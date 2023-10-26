import pytest
import torch

from aimet_ml.features.textual.transformers import TransformerFeatureExtractor


@pytest.fixture
def feature_extractor() -> TransformerFeatureExtractor:
    """
    Fixture for creating an instance of TransformerFeatureExtractor for testing.

    Returns:
        TransformerFeatureExtractor: An instance of the feature extractor for testing.
    """
    return TransformerFeatureExtractor(
        model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        num_emb_layers=2,
        max_length=128,
        device="cuda:0",
    )


def test_device(feature_extractor: TransformerFeatureExtractor) -> None:
    """
    Test the device of the model in the feature extractor.

    Args:
        feature_extractor (TransformerFeatureExtractor): The feature extractor instance for testing.
    """
    if torch.cuda.is_available():
        assert "cuda" == feature_extractor.model.device.type

        cpu_feature_extractor = TransformerFeatureExtractor(
            model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
            num_emb_layers=2,
            max_length=128,
            device="cpu",
        )
        assert "cpu" == cpu_feature_extractor.model.device.type

    else:
        assert "cpu" == feature_extractor.model.device.type


def test_tokenize(feature_extractor: TransformerFeatureExtractor) -> None:
    """
    Test the tokenize method of the feature extractor.

    Args:
        feature_extractor (TransformerFeatureExtractor): The feature extractor instance for testing.
    """
    text = "This is a test sentence."
    tokenized = feature_extractor.tokenize(text)

    assert isinstance(tokenized, dict)
    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    assert isinstance(tokenized["input_ids"], torch.Tensor)
    assert isinstance(tokenized["attention_mask"], torch.Tensor)
    assert tokenized["input_ids"].shape == (1, feature_extractor.max_length)
    assert tokenized["attention_mask"].shape == (1, feature_extractor.max_length)


def test_extract_features(feature_extractor: TransformerFeatureExtractor) -> None:
    """
    Test the extract_features method of the feature extractor.

    Args:
        feature_extractor (TransformerFeatureExtractor): The feature extractor instance for testing.
    """
    texts = ["This is sentence 1.", "Another sentence here."]
    features_df = feature_extractor.extract_features(texts)

    assert isinstance(features_df, torch.Tensor)
    assert features_df.shape == (
        2,
        feature_extractor.model.config.hidden_size,
    )


def test_extract_single_text(feature_extractor: TransformerFeatureExtractor) -> None:
    """
    Test the edge case for extract_features method with a single text.

    Args:
        feature_extractor (TransformerFeatureExtractor): The feature extractor instance for testing.
    """
    text = "Only one sentence."
    features_df = feature_extractor.extract_features(text)

    assert isinstance(features_df, torch.Tensor)
    assert features_df.shape == (
        1,
        feature_extractor.model.config.hidden_size,
    )


if __name__ == "__main__":
    pytest.main()
