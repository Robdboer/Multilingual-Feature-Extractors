# ASReview distiluse-base-multilingual-cased-v2 feature extractor

This extension to ASReview implements a multilingual feature extractor algorithm, allowing for the analysis of records in multiple languages.
The following feature extractor is implemented:

- sentence-transformers/distiluse-base-multilingual-cased-v2 [Source](sentence-transformers/distiluse-base-multilingual-cased-v2)

## Getting started

This model depends on Sentence-Transformers. Install it with:

```bash
pip install sentence-transformers
```

Install the multilingual feature extractors with:


```bash
pip install git+https://github.com/Robdboer/multilingual-feature-extractors.git
```

## Usage

The feature extractor are defined in
[`asreviewcontrib/models/distiluse_base_multilingual_cased_v2`](asreviewcontrib/models/distiluse_base_multilingual_cased_v2) and can be used in a simulation.

```bash
asreview simulate benchmark:van_de_Schoot_2017 -e distiluse -m svm
```

> Please note that, as with all sentence transformers, this model produces negative vector values. Consequently, it is not compatible with Naive Bayes classifiers, which require non-negative feature values.


## License

MIT license
