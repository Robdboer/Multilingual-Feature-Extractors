# ASReview bert-base-multilingual-cased feature extractor

This extension to ASReview implements a multilingual feature extractor algorithm, allowing for the analysis of records in multiple languages.
The following feature extractor is implemented:

- transformers/bert-base-multilingual-cased [Source](https://huggingface.co/bert-base-multilingual-cased)

## Getting started

This model depends on Transformers. Install it with:

```bash
pip install transformers
```

Install the multilingual feature extractors with:


```bash
pip install git+https://github.com/Robdboer/multilingual-feature-extractors.git
```

## Usage

The feature extractor are defined in
[`asreviewcontrib/models/bert-base-multilingual-cased`](asreviewcontrib/models/bert-base-multilingual-cased) and can be used in a simulation.

```bash
asreview simulate benchmark:van_de_Schoot_2017 -e mbert -m svm
```

> Please note that, as with all transformers, this model produces negative vector values. Consequently, it is not compatible with Naive Bayes classifiers, which require non-negative feature values.


## License

MIT license
