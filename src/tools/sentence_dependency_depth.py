import spacy
from langchain_core.tools import tool

nlp = spacy.load("en_core_web_sm")


@tool
def sentence_dependency_depth(text: str) -> float:
    """Calculates Average syntactical sentence dependency depth."""
    def get_depth(token):
        """recrusive function to determine depth of token."""
        if not list(token.children):
            return 1
        return 1 + max(get_depth(child) for child in token.children)

    doc = nlp(text)
    depths = []
    for sent in doc.sents:
        depths.append(get_depth(sent.root))

    return sum(depths) / len(depths) if depths else 0.0
