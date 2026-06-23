from src.data_loader import map_star_to_sentiment
from src.preprocessing import clean_text, lowercase_text, preprocess_text, remove_stopwords


def test_lowercase_text():
    assert lowercase_text("Great PRODUCT") == "great product"


def test_clean_text_removes_punctuation_and_extra_spaces():
    assert clean_text("Great!!!   Product, works.") == "Great Product works"


def test_remove_stopwords_removes_common_words():
    assert remove_stopwords("this is a great product") == "great product"


def test_preprocess_text_returns_string():
    result = preprocess_text("This PRODUCT works very well!!!")
    assert isinstance(result, str)
    assert "product" in result


def test_map_star_to_sentiment():
    assert map_star_to_sentiment(1) == "negative"
    assert map_star_to_sentiment(2) == "negative"
    assert map_star_to_sentiment(3) == "neutral"
    assert map_star_to_sentiment(4) == "positive"
    assert map_star_to_sentiment(5) == "positive"

