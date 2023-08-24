import torch
import pandas as pd
from typing import Union, List

from transformers import AutoModel, AutoTokenizer


class TransformerFeatureExtractor:
    def __init__(
        self,
        model_name: str,
        num_emb_layers: int = 4,
        max_length: int = 512,
        device: Union[str, torch.device] = "cuda:0",
    ):
        """
        Initializes the TransformerFeatureExtractor.

        Parameters:
            model_name (str): The name or path of the pre-trained transformer model.
            num_emb_layers (int, optional): Number of layers to use for feature extraction. Default is 4.
            max_length (int, optional): Maximum length of input text for tokenization. Default is 512.
            device (str or torch.device, optional): Device to use for computation ('cuda:0', 'cpu', etc.). Default is 'cuda:0' if available, else 'cpu'.
        """

        if not torch.cuda.is_available():
            device = "cpu"

        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_emb_layers = num_emb_layers
        self.max_length = max_length

    def extract_features(self, texts: Union[str, List[str]]) -> pd.DataFrame:
        """
        Extracts features from input texts using transformer embeddings.

        Parameters:
            texts (str or list): Input text or list of texts for feature extraction.

        Returns:
            pd.DataFrame: A DataFrame containing extracted features for each input text.
        """
        self.model.eval()

        if isinstance(texts, str):
            texts = [texts]

        input_ids, attention_masks = [], []
        for text in texts:
            tokenized_output = self.tokenize(text)
            input_ids.append(tokenized_output["input_ids"])
            attention_masks.append(tokenized_output["attention_mask"])
        input_ids = torch.cat(input_ids, dim=0).to(self.model.device)
        attention_masks = torch.cat(attention_masks, dim=0).to(self.model.device)

        with torch.no_grad():
            hidden_states = self.model(
                input_ids, attention_mask=attention_masks, output_hidden_states=True
            )["hidden_states"]

        embeddings = sum(
            hidden_states[-i][:, 0, :] for i in range(1, self.num_emb_layers + 1)
        )
        embeddings = embeddings.detach().cpu().numpy()

        emb_df = pd.DataFrame(embeddings)
        emb_df.index = texts

        return emb_df

    def tokenize(self, text: str) -> dict:
        """
        Tokenizes input text using the transformer's tokenizer.

        Parameters:
            text (str): Input text to be tokenized.

        Returns:
            dict: Dictionary containing tokenized input with attention mask.
        """
        tokenized_output = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return tokenized_output.data
