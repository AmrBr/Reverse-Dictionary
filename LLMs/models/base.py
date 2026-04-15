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

SYSTEM_PROMPT = """أنت قاموس عكسي للغة العربية. بناءً على التعريف المعطى، اذكر أفضل 5 كلمات عربية تناسب هذا التعريف.

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
5. كلمة"""

SYSTEM_PROMPT_ENG = """You are an Arabic reverse dictionary. Based on the given definition, list the top 5 Arabic words that best fit this definition.
Rules:
- Only return a numbered list from 1 to 5
- Each answer should be a single Arabic word or phrase
- Rank from most likely to least likely
- Do not write any additional explanation or text
- Do not repeat the same word
- Do not write any introduction, explanation, or conclusion. Just return the numbered list directly, each line containing one word or short phrase.
- Answer in Arabic Only.
Example of the desired format:
1. Word
2. Word
3. Word
4. Word
5. Word"""

USER_PROMPT = """{examples}

# التعريف: {definition}

"""


class BaseModel(ABC):
    """All model backends must implement this interface."""

    def build_prompt(self, definition: str, examples: str = "") -> str:
        return PROMPT_TEMPLATE.format(definition=definition, examples=examples)
    
    def build_system_and_user_prompts(self, definition: str, examples: str = "") -> tuple[str, str]:
        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT.format(definition=definition, examples=examples)
        return system_prompt, user_prompt

    @abstractmethod
    def query(self, definition: str) -> str:
        """Given a definition string, return the raw model output."""
        ...