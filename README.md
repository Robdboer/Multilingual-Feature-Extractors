# ASReview multilingual feature extractors

This extension to ASReview implements several multilingual feature extractor algorithms, allowing for the analysis of records in multiple languages.
The following sentence transformers are currently implemented:

- sentence-transformers/distiluse-base-multilingual-cased-v2 [Source](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)

- sentence-transformers/paraphrase-multilingual-mpnet-base-v2 [Source](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)

- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 [Source](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

- sentence-transformers/stsb-xlm-r-multilingual [Source](sentence-transformers/stsb-xlm-r-multilingual)

and more are in the pipeline.



## Getting started

This model depends on Sentence-Transformers. Install it with:

```bash
pip install sentence-transformers
```

Install the multilingual feature extractors with:


```bash
pip install git+https://github.com/Robdboer/multilingual-sentence-transformers.git
```

## Usage

The feature extractors are defined in
[`asreviewcontrib/models`](asreviewcontrib/models) and can be used in a simulation.

```bash
asreview simulate benchmark:van_de_Schoot_2017 -e [Extractor_Model] -m svm
```

Choose an Extractor Model out of the following:

- minilm (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

> Please note that, as with all sentence transformers, this model produces negative vector values. Consequently, it is not compatible with Naive Bayes classifiers, which require non-negative feature values.


## License

MIT license
