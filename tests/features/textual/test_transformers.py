import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


import pytest
import torch
import pandas as pd
from aimet_ml.features.textual.transformers import TransformerFeatureExtractor


# Create an instance of the TransformerFeatureExtractor for testing
@pytest.fixture
def feature_extractor():
    return TransformerFeatureExtractor(
        model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        num_emb_layers=2,
        max_length=128,
        device="cuda:0",
    )


# Test device of the model
def test_device(feature_extractor):

    if torch.cuda.is_available():
        assert "cuda" == feature_extractor.model.device.type
    else:
        assert "cpu" == feature_extractor.model.device.type

    cpu_feature_extractor = TransformerFeatureExtractor(
        model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        num_emb_layers=2,
        max_length=128,
        device="cpu",
    )
    assert "cpu" == cpu_feature_extractor.model.device.type


# Test tokenize method
def test_tokenize(feature_extractor):
    text = "This is a test sentence."
    tokenized = feature_extractor.tokenize(text)

    assert isinstance(tokenized, dict)
    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    assert isinstance(tokenized["input_ids"], torch.Tensor)
    assert isinstance(tokenized["attention_mask"], torch.Tensor)
    assert tokenized["input_ids"].shape == (1, feature_extractor.max_length)
    assert tokenized["attention_mask"].shape == (1, feature_extractor.max_length)


# Test extract_features method
def test_extract_features(feature_extractor):
    texts = ["This is sentence 1.", "Another sentence here."]
    features_df = feature_extractor.extract_features(texts)

    assert isinstance(features_df, pd.DataFrame)
    assert features_df.shape == (
        2,
        feature_extractor.model.config.hidden_size,
    )


# Test edge case for extract_features with a single text
def test_extract_single_text(feature_extractor):
    text = "Only one sentence."
    features_df = feature_extractor.extract_features(text)

    assert isinstance(features_df, pd.DataFrame)
    assert features_df.shape == (
        1,
        feature_extractor.model.config.hidden_size,
    )


if __name__ == "__main__":
    pytest.main()
