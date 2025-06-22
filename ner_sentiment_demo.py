import spacy
from spacy.tokens import Span

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Sample text for demonstration
text = "I love my new Apple iPhone! The camera is amazing, but Samsung's Galaxy series is also great. However, I had a bad experience with a cheap charger from BrandX."

doc = nlp(text)

# Extract product and brand entities (using ORG and PRODUCT labels)
entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]

print("Extracted Entities (Product/Brand):")
for ent_text, ent_label in entities:
    print(f"- {ent_text} ({ent_label})")

# Simple rule-based sentiment analysis
def rule_based_sentiment(text):
    positive_words = {"love", "amazing", "great", "good", "excellent", "awesome", "fantastic"}
    negative_words = {"bad", "terrible", "awful", "poor", "worst", "disappointing", "cheap"}
    
    text_lower = text.lower()
    pos_count = sum(word in text_lower for word in positive_words)
    neg_count = sum(word in text_lower for word in negative_words)
    
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    else:
        return "Neutral"

sentiment = rule_based_sentiment(text)
print(f"\nSentiment: {sentiment}") 