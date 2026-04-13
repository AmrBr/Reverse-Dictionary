import re


# Matches both Western (1. 2) and Arabic-Indic numerals (١. ٢)
_NUMERAL = r"[\d\u0660-\u0669]+"

# Arabic word characters: letters + diacritics (tashkeel) + tatweel + spaces for phrases
_ARABIC_WORD = r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s\-]+"

# Latin fallback (in case model still responds in English)
_LATIN_WORD = r"[a-zA-Z\-]+"

_LINE_PATTERN = re.compile(
    rf"^\s*{_NUMERAL}[\.\)\-\s]+\s*({_ARABIC_WORD}|{_LATIN_WORD})"
)


def parse_response(text: str, debug: bool = False) -> list[str]:
    """
    Extract a ranked list of words from raw model output.

    Handles:
    - Western numerals:       1. كلمة  /  1) كلمة
    - Arabic-Indic numerals:  ١. كلمة
    - Diacritics (tashkeel):  كَلِمَة
    - Multi-word labels:      رقص حديث
    - Latin fallback:         word

    Returns up to 5 stripped entries.
    """
    words = []

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        match = _LINE_PATTERN.match(line)
        if match:
            word = match.group(1).strip()
            # Collapse internal whitespace for multi-word phrases
            word = re.sub(r"\s+", " ", word)
            if word:
                words.append(word)
        elif debug:
            print(f"[parser] unmatched line: {repr(line)}")

    return words[:5]