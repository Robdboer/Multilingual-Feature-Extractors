# ASReview labse feature extractor

This extension to ASReview implements a multilingual feature extractor algorithm, allowing for the analysis of records in multiple languages.
The following feature extractor is implemented:

- sentence-transformers/labse [Source](sentence-transformers/labse)

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
[`asreviewcontrib/models/labse`](asreviewcontrib/models/labse) and can be used in a simulation.

```bash
asreview simulate benchmark:van_de_Schoot_2017 -e labse -m svm
```

> Please note that, as with all sentence transformers, this model produces negative vector values. Consequently, it is not compatible with Naive Bayes classifiers, which require non-negative feature values.


## License

MIT license
