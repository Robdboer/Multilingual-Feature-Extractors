import tensorflow_hub as hub
import numpy as np
import tensorflow_text
from asreview.models.feature_extraction.base import BaseFeatureExtraction


class muse(BaseFeatureExtraction):
    """
    Multilingual Sentence Transformer feature extraction technique using
    the 'muse' model.

    This class inherits from the BaseFeatureExtraction class provided by the
    ASReview package and implements the transform method for encoding texts
    using the multilingual SentenceTransformer model.
    """

    name = "muse"
    label = "muse"

    def transform(self, texts):
        """
        Encode the given texts using the SentenceTransformer model.

        Args:
        texts (List[str]): A list of text strings to be encoded.

        Returns:
        numpy.ndarray: A 2D array containing the encoded text embeddings.
        """

        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

        embeddings = []
        for text in texts:
            embeddings.append(embed(text))

        return embeddings