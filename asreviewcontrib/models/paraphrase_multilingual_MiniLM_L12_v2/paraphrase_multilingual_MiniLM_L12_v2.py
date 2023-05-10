from sentence_transformers import SentenceTransformer
from asreview.models.feature_extraction.base import BaseFeatureExtraction


class paraphrasemultilingualMiniLML12v2(BaseFeatureExtraction):
    """
    Multilingual Sentence Transformer feature extraction technique using
    the 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' model.

    This class inherits from the BaseFeatureExtraction class provided by the
    ASReview package and implements the transform method for encoding texts
    using the multilingual SentenceTransformer model.
    """

    name = "minilm"
    label = "paraphrase-multilingual-MiniLM-L12-v2"

    def transform(self, texts):
        """
        Encode the given texts using the SentenceTransformer model.

        Args:
        texts (List[str]): A list of text strings to be encoded.

        Returns:
        numpy.ndarray: A 2D array containing the encoded text embeddings.
        """

        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        # Wrap texts with tqdm for progress bar
        print(
            "Encoding texts using the paraphrase-multilingual-MiniLM-L12-v2 model, this may take a while..."
        )
        return model.encode(texts, show_progress_bar=True)
