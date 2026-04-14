from abc import ABC, abstractmethod

PROMPT_TEMPLATE = """أنت قاموس عكسي للغة العربية. بناءً على التعريف المعطى، اذكر أفضل 5 كلمات عربية تناسب هذا التعريف.

القواعد:
- أعد فقط قائمة مرقمة من 1 إلى 5
- كل إجابة يجب أن تكون كلمة أو عبارة عربية واحدة فقط
- رتب من الأكثر احتمالاً إلى الأقل
- لا تكتب أي شرح أو نص إضافي
- لا تكرر نفس الكلمة
- لا تكتب أي مقدمة أو شرح أو خاتمة.
أعد فقط القائمة المرقمة مباشرة، كل سطر فيه كلمة واحدة أو عبارة قصيرة.
مثال على الشكل المطلوب:
1. كلمة
2. كلمة
3. كلمة
4. كلمة
5. كلمة

{examples}

# التعريف: {definition}

أفضل 5 كلمات:"""


class BaseModel():
    """All model backends must implement this interface."""

    def build_prompt(self, definition: str, examples: str = "") -> str:
        return PROMPT_TEMPLATE.format(definition=definition, examples=examples)

    @abstractmethod
    def query(self, definition: str) -> str:
        """Given a definition string, return the raw model output."""
        ...