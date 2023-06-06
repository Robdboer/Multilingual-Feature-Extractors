import numpy as np
from laserembeddings import Laser
from langdetect import detect
from asreview.models.feature_extraction.base import BaseFeatureExtraction


class laser(BaseFeatureExtraction):
    """
    Multilingual Sentence Transformer feature extraction technique using
    the 'laser' model.

    This class inherits from the BaseFeatureExtraction class provided by the
    ASReview package and implements the transform method for encoding texts
    using the multilingual SentenceTransformer model.
    """

    name = "laser"
    label = "laser"

    def transform(self, texts):
        """
        Encode the given texts using the SentenceTransformer model.

        Args:
        texts (List[str]): A list of text strings to be encoded.

        Returns:
        numpy.ndarray: A 2D array containing the encoded text embeddings.
        """

        laser = Laser()

        num_texts = len(texts)
        print('Setting up multilingual model with laser... This could take a while')

        embeddings = []
        count = 1
        for text in texts:
            language = text.apply(detect)
            print(language, end='\r')
            print((count/num_texts)*100, '%', end='\r')
            embeddings.append(laser.embed_sentences(text, lang=language))
            count += 1

        return np.concatenate(embeddings)