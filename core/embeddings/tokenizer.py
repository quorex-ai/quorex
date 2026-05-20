import re
import string

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "it", "its", "this", "that", "i", "you",
    "he", "she", "we", "they", "not", "no", "so", "as", "if", "then",
}

class Tokenizer:
    def __init__(self, min_length: int = 2, remove_stopwords: bool = True):
        self.min_length = min_length
        self.remove_stopwords = remove_stopwords

    def tokenize(self, text: str) -> list[str]:
        """
        Transforms raw text into a list of clean tokens.

        Steps:
        1. Lowercase
        2. Remove punctuation
        3. Split on whitespace
        4. Filter short tokens
        5. Remove stopwords (optional)
        """
        if not text or not isinstance(text, str):
            return []

        # Lowercase
        text = text.lower()
        # Replace punctuation with spaces
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Split into tokens
        tokens = text.split()
        # Filter short tokens
        tokens = [t for t in tokens if len(t) >= self.min_length]
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]

        return tokens
    
    ATOMIC_FIELDS = {"action", "event", "type", "eventType", "event_type"}

    def tokenize_event(self, event: dict) -> list[str]:
        """ 
        Tokenizes a Quorex user event dict.

        Rules :
        - ATOMIC_FIELDS (action, type...) -> kept as single tokens, never split "viewed_pricing" stays "viewed_pricing"
        - All other string fields -> normal tokenization "pro dashboard" -> ["pro", "dashboard"]
        - Nested metadata -> normal tokenization on values

        Example :
        {
            "action": "viewed_pricing",      → ["viewed_pricing"]
            "userId": "user_123",            → ["user", "123"]
            "metadata": {
                "plan": "pro",               → ["pro"]
                "source": "dashboard"        → ["dashboard"]
            }
        }
        -> ["viewed_pricing", "user", "123", "pro", "dashboard"]
        """
        tokens = []
 
        for key, value in event.items():
            if isinstance(value, str):
                if key in self.ATOMIC_FIELDS:
                    # Keep as atomic token — just lowercase, no split
                    tokens.append(value.lower())
                else:
                    tokens.extend(self.tokenize(value))
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, str):
                        if k in self.ATOMIC_FIELDS:
                            tokens.append(v.lower())
                        else:
                            tokens.extend(self.tokenize(v))
 
        return tokens
    
if __name__ == "__main__":
    t = Tokenizer()

    print(t.tokenize("The user viewed the pricing page from the dashboard."))

    print(t.tokenize_event({
        "action": "viewed_pricing",
        "userId": "user_123",
        "metadata": { "plan": "pro", "sources": "dashboard"}
    }))