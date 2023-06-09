import numpy as np
from asreview.models.feature_extraction.base import BaseFeatureExtraction
import glob
from pathlib import Path
import os


class readfm(BaseFeatureExtraction):
    """
    Multilingual Sentence Transformer feature extraction technique using
    the 'readfm' model.

    This class inherits from the BaseFeatureExtraction class provided by the
    ASReview package and implements the transform method for encoding texts
    using the multilingual model.
    """

    name = "readfm"
    label = "readfm"

    def transform(self, texts):
        """
        Encode the given texts using the bert model.
        Args:
        texts (List[str]): A list of text strings to be encoded.
        Returns:
        numpy.ndarray: A 2D array containing the encoded text embeddings.
        """
 
        print('Trying to read matrix...')

        path = Path(glob.glob('*.npy')[0])
        feature_matrix = np.load(path)
        os.remove(path)
        return feature_matrix