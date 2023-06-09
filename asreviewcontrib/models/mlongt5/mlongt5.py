from transformers import T5Tokenizer, LongT5Model
import numpy as np
from asreview.models.feature_extraction.base import BaseFeatureExtraction
import glob
from pathlib import Path
import torch
import os


class mlongt5(BaseFeatureExtraction):
    """
    Multilingual Sentence Transformer feature extraction technique using
    the 'mlongt5' model.

    This class inherits from the BaseFeatureExtraction class provided by the
    ASReview package and implements the transform method for encoding texts
    using the multilingual model.
    """

    name = "mlongt5"
    label = "mlongt5"

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

        tokenizer = T5Tokenizer.from_pretrained("agemagician/mlong-t5-tglobal-base")
        model = LongT5Model.from_pretrained("agemagician/mlong-t5-tglobal-base")
        encoded_input = tokenizer(texts.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=1024)

        embeddings = []
        batch_size = 10
        num_texts = len(texts)
        for i in range(0, num_texts, batch_size):
            with torch.no_grad():
                start_idx = i
                end_idx = min(i + batch_size, num_texts)
                batch_input = {k:v[start_idx:end_idx] for k, v in encoded_input.items()}
                output = model.encoder(**batch_input)
                batch_embeddings = output.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(batch_embeddings)
                percent_complete = int((end_idx / num_texts) * 100)
                print(f"Processing texts: {percent_complete}%", end="\r")

        print('Saving matrix...')
        np.save(Path("feature_matrix_" + str(len(feature_matrices)+1) + ".npy"), np.concatenate(embeddings))

        return np.concatenate(embeddings)