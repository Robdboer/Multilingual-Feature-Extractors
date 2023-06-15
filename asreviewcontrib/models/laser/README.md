# ASReview laser feature extractor

This extension to ASReview implements a multilingual feature extractor algorithm, allowing for the analysis of records in multiple languages.
The following feature extractor is implemented:

- laser [Source](https://pypi.org/project/laserembeddings/)

## Getting started

This model depends on laserembeddings. Install it with:

```bash
pip install laserembeddings
pip install laserembeddings[zh]
```

Install the multilingual feature extractors with:


```bash
pip install git+https://github.com/Robdboer/multilingual-feature-extractors.git
```

## Usage

The feature extractor are defined in
[`asreviewcontrib/models/laser`](asreviewcontrib/models/laser) and can be used in a simulation.

```bash
asreview simulate benchmark:van_de_Schoot_2017 -e laser -m svm
```

> Please note that this model produces negative vector values. Consequently, it is not compatible with Naive Bayes classifiers, which require non-negative feature values.


## License

MIT license
