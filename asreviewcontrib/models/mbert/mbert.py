import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from asreview.models.feature_extraction.base import BaseFeatureExtraction
import glob
from pathlib import Path

class mbert(BaseFeatureExtraction):
    """
    Multilingual Bert Base feature extraction technique using
    the 'bert-base-multilingual-cased' model.
    This class inherits from the BaseFeatureExtraction class provided by the
    ASReview package and implements the transform method for encoding texts
    using the multilingual SentenceTransformer model.
    """

    name = "mbert"
    label = "mbert"

    def transform(self, texts):
        """
        Encode the given texts using the bert model.
        Args:
        texts (List[str]): A list of text strings to be encoded.
        Returns:
        numpy.ndarray: A 2D array containing the encoded text embeddings.
        """
 
        print('Setting up multilingual model... This could take a while')

        feature_matrices = glob.glob('*.npy')

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained("bert-base-multilingual-cased")
        encoded_input = tokenizer(texts.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=512)

        embeddings = []
        batch_size = 10
        num_texts = len(texts)
        for i in range(0, num_texts, batch_size):
            with torch.no_grad():
                start_idx = i
                end_idx = min(i + batch_size, num_texts)
                batch_input = {k:v[start_idx:end_idx] for k, v in encoded_input.items()}
                output = model(**batch_input)
                batch_embeddings = output.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(batch_embeddings)
                percent_complete = int((end_idx / num_texts) * 100)
                print(f"Processing texts: {percent_complete}%", end="\r")

        print('Saving matrix...')
        np.save(Path("feature_matrix_" + str(len(feature_matrices)+1) + ".npy"), np.concatenate(embeddings))

        return np.concatenate(embeddings)