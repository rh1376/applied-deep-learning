from __future__ import annotations

from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from src.rag_helpdesk.retriever import Retriever

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_CONTEXT_CHARS = 6000
FALLBACK_FA = "اطلاعات کافی در اسناد موجود نیست."
# اگر بهترین امتیاز از این بالاتر بود یعنی retrieval مرتبطه
RETRIEVAL_STRONG_SCORE = 0.45
# اگر میانگین چند تای اول از این بالاتر بود یعنی کلیت context خوبه
RETRIEVAL_STRONG_MEAN_SCORE = 0.35
# برای میانگین‌گیری چند نتیجه اول
CONFIDENCE_TOP_N = 3

def _retrieval_confidence(retrieved: list[dict], top_n: int = CONFIDENCE_TOP_N) -> tuple[float, float]:
    """
    برمی‌گرداند:
    - best_score: بیشترین شباهت
    - mean_top: میانگین top_n اول
    """
    if not retrieved:
        return 0.0, 0.0

    scores = [float(x.get("score", 0.0)) for x in retrieved if x is not None]
    if not scores:
        return 0.0, 0.0

    best = max(scores)
    n = max(1, min(top_n, len(scores)))
    mean_top = sum(scores[:n]) / n
    return best, mean_top

def format_context(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""

    blocks: list[str] = []
    for idx, item in enumerate(items, start=1):
        source_path = str(item.get("metadata", {}).get("source_path", ""))
        chunk_id = str(item.get("chunk_id", ""))
        text = str(item.get("text", "")).strip()
        blocks.append(f"[{idx}] source={source_path} chunk_id={chunk_id}\n{text}")

    return "\n\n".join(blocks)

class RAGEngine:
    def __init__(self, retriever: Retriever, llm_model_name: str, max_new_tokens: int = 256) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")

        model_name = llm_model_name or DEFAULT_LLM_MODEL

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for generation. Install it with: "
                "pip install transformers torch"
            ) from exc

        self.retriever = retriever
        self.llm_model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer = AutoTokenizer.from_pretrained("models/qwen-3b")
        self._model = AutoModelForCausalLM.from_pretrained(
            "models/qwen-3b",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._generator = pipeline(
            task="text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            return_full_text=False,
        )

    def _build_retry_prompt(self, query: str, context_items: list[dict[str, Any]]) -> str:
        system_msg = (
            "تو یک دستیار Helpdesk هستی.\n"
            "پاسخ باید فقط از Context استخراج شود.\n"
            "اگر در Context اطلاعات مرتبط وجود دارد، حق نداری جمله «اطلاعات کافی...» را بنویسی.\n"
            "پاسخ را خیلی کوتاه و مستقیم بده.\n"
            "در پایان فقط شماره منابعی که استفاده کردی را در Citations بیاور."
        )

        user_msg = (
            f"Query:\n{query.strip()}\n\n"
            f"Context:\n{format_context(context_items)}\n\n"
            "Output:\n"
            "Answer: <پاسخ مستقیم از Context>\n"
            "Citations:\n"
            "- [n]\n"
        )

        return self._tokenizer.apply_chat_template(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _select_context_items(self, retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        total_chars = 0

        for item in retrieved:
            text = str(item.get("text", ""))
            if not text:
                continue

            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining <= 0:
                break

            if len(text) > remaining:
                item_copy = dict(item)
                item_copy["text"] = text[:remaining].rstrip()
                if item_copy["text"]:
                    selected.append(item_copy)
                    total_chars += len(item_copy["text"])
                break

            selected.append(item)
            total_chars += len(text)

        return selected

    def _build_chat_prompt(self, query: str, context_items: list[dict[str, Any]]) -> str:
        system_msg = (
            "شما یک دستیار Helpdesk هستید.\n"
            "فقط و فقط با استفاده از متن «Context» پاسخ بده و هیچ چیز از خودت اضافه نکن.\n"
            "اگر Context کافی نبود، دقیقاً همین جمله را بگو: «اطلاعات کافی در اسناد موجود نیست.»\n"
            "در پایان حتماً بخش «Citations» را بنویس و شماره آیتم‌های Context را مثل [1]، [2] ذکر کن.\n"
            "پاسخ را فارسی بنویس."
        )

        user_msg = (
            f"سؤال:\n{query.strip()}\n\n"
            f"Context:\n{format_context(context_items)}\n\n"
            "خروجی را دقیقاً با این قالب بده:\n"
            "Answer: ...\n"
            "Citations:\n"
            "- [n] ..."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def answer(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        retrieved = self.retriever.retrieve(query)
        context_items = self._select_context_items(retrieved)
        context_texts = [str(item.get("text", "")) for item in context_items]

        citations = [
            {
                "score": float(item.get("score", 0.0)),
                "source_path": str(item.get("metadata", {}).get("source_path", "")),
                "chunk_id": str(item.get("chunk_id", "")),
                "doc_id": str(item.get("doc_id", "")),
            }
            for item in context_items
        ]

        if not context_items:
            return {
                "answer": "اطلاعات کافی در اسناد موجود نیست.",
                "citations": citations,
                "context_used": context_texts,
            }
        
        prompt = self._build_chat_prompt(query=query, context_items=context_items)
        outputs = self._generator(prompt, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=0.0)

        answer_text = ""
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict):
                answer_text = str(first.get("generated_text", "")).strip()

        if not answer_text:
            answer_text = "اطلاعات کافی در اسناد موجود نیست."

        fallback = FALLBACK_FA

        best_score, mean_top = _retrieval_confidence(retrieved, top_n=CONFIDENCE_TOP_N)

        retrieval_is_strong = (
            len(context_items) > 0 and
            (best_score >= RETRIEVAL_STRONG_SCORE or mean_top >= RETRIEVAL_STRONG_MEAN_SCORE)
        )

        # اگر retrieval قویه ولی مدل fallback داده، یک بار retry با prompt ساده‌تر
        if retrieval_is_strong and answer_text.strip().startswith(fallback):
            retry_prompt = self._build_retry_prompt(query=query, context_items=context_items)
            outputs2 = self._generator(
                retry_prompt,
                max_new_tokens=min(self.max_new_tokens, 192),
                do_sample=False,
                temperature=0.0,
            )

            retry_text = ""
            if isinstance(outputs2, list) and outputs2:
                first2 = outputs2[0]
                if isinstance(first2, dict):
                    retry_text = str(first2.get("generated_text", "")).strip()

            # اگر retry خروجی معنادار داد، جایگزین کن
            if retry_text and not retry_text.strip().startswith(fallback):
                answer_text = retry_text
        
        return {
            "answer": answer_text,
            "citations": citations,
            "context_used": context_texts,
        }