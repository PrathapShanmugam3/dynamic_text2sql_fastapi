import re

STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with",
    "is", "are", "was", "were", "get", "show", "list", "find", "all",
    "me", "please", "what", "which", "how", "many", "much", "give"
}

def _tokenize(text: str) -> set:
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}

def filter_relevant_schema(schema: dict, question: str, max_tables: int = 6) -> dict:
    question_words = _tokenize(question)
    if not question_words:
        return schema

    scored = []
    for table_name, columns in schema.items():
        table_words = _tokenize(table_name)
        column_words = set()
        for col in columns:
            column_words |= _tokenize(col["name"])

        score = 0
        for qw in question_words:
            if any(qw in tw or tw in qw for tw in table_words):
                score += 3
            if any(qw in cw or cw in qw for cw in column_words):
                score += 1

        scored.append((score, table_name))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    relevant = [name for score, name in scored if score > 0][:max_tables]

    if not relevant:
        relevant = [name for _, name in scored[:max_tables]]

    return {name: schema[name] for name in relevant}
