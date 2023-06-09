# ASReview stsb-xlm-r-multilingual feature extractor

This extension to ASReview implements a multilingual feature extractor algorithm, allowing for the analysis of records in multiple languages.
The following feature extractor is implemented:

- sentence-transformers/stsb-xlm-r-multilingual [Source](sentence-transformers/stsb-xlm-r-multilingual)

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
[`asreviewcontrib/models/stsb_xlm_r_multilingual`](asreviewcontrib/models/stsb_xlm_r_multilingual) and can be used in a simulation.

```bash
asreview simulate benchmark:van_de_Schoot_2017 -e stsb -m svm
```

> Please note that, as with all sentence transformers, this model produces negative vector values. Consequently, it is not compatible with Naive Bayes classifiers, which require non-negative feature values.


## License

MIT license
