import numpy as np
from laserembeddings import Laser
from langdetect import detect
from asreview.models.feature_extraction.base import BaseFeatureExtraction
from pathlib import Path
import glob


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

        feature_matrices = glob.glob('*.npy')

        # load model if exists
        if len(feature_matrices) > 0 and len(feature_matrices) < 5:
            cache_fp = Path("feature_matrix_" + str(len(feature_matrices)) + ".npy")
            print('Loading matrix from: ' + str(cache_fp))
            matrix = np.load(cache_fp)
            np.save(Path("feature_matrix_" + str(len(feature_matrices)+1) + ".npy"), matrix)
            return matrix
        else:
            embeddings = []
            count = 0
            for text in texts:
                try:
                    lang = detect(text)[:2]
                except:
                    lang = 'en'

                print(lang, (count/num_texts)*100, '%', end='\r')
                embeddings.append(laser.embed_sentences(text, lang=lang))
                count += 1

            print('Saving matrix...')
            np.save(Path("feature_matrix_" + str(1) + ".npy"), np.concatenate(embeddings))

            return np.concatenate(embeddings)