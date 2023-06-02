import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece
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

        model_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/1"

        # Graph set up.
        g = tf.Graph()
        with g.as_default():
          text_input = tf.placeholder(dtype=tf.string, shape=[None])
          embed = hub.Module(model_url)
          embedded_text = embed(text_input)
          init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()

        # Initialize session.
        session = tf.Session(graph=g)
        session.run(init_op)
        # Compute embeddings.
        embeddings = []
        for text in texts:
        	embeddings.append(session.run(embedded_text, feed_dict={text_input: text}))

        return embeddings
