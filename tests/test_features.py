import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from spam_classifier.data import DEFAULT_DATA_PATH, load_dataset, split_dataset
from spam_classifier.features import FEATURE_NAMES, build_vectorizer
from tests.conftest import needs_real_data


def test_feature_names():
    assert FEATURE_NAMES == ("word", "char", "word+char")


def test_word_vectorizer_matches_the_notebook_configuration():
    vec = build_vectorizer("word")
    assert isinstance(vec, TfidfVectorizer)
    assert vec.lowercase is True
    assert vec.stop_words == "english"
    assert vec.analyzer == "word"
    assert vec.ngram_range == (1, 1)


def test_char_vectorizer_uses_character_ngrams():
    vec = build_vectorizer("char")
    assert vec.analyzer == "char_wb"
    assert vec.ngram_range == (2, 5)
    assert vec.sublinear_tf is True


def test_word_plus_char_produces_both_kinds_of_features(toy_messages):
    texts, _ = toy_messages
    matrix = build_vectorizer("word+char").fit_transform(texts)
    word = build_vectorizer("word").fit_transform(texts)
    char = build_vectorizer("char").fit_transform(texts)
    assert matrix.shape[1] == word.shape[1] + char.shape[1]


def test_normalize_flag_changes_the_text_the_vectorizer_sees(toy_messages):
    texts, _ = toy_messages
    plain = build_vectorizer("word").fit(texts).get_feature_names_out()
    normalised = build_vectorizer("word", normalize=True).fit(texts).get_feature_names_out()
    assert "urltoken" in normalised and "urltoken" not in plain


def test_unknown_feature_kind_is_rejected():
    with pytest.raises(ValueError, match="trigram"):
        build_vectorizer("trigram")


@needs_real_data
def test_word_vocabulary_size_on_the_training_split_is_6839():
    df = load_dataset(DEFAULT_DATA_PATH)
    x_train, _, _, _ = split_dataset(df)
    vec = build_vectorizer("word").fit(x_train.tolist())
    assert len(vec.vocabulary_) == 6839
