# ASReview mlong-t5-tglobal-base feature extractor

This extension to ASReview implements a multilingual feature extractor algorithm, allowing for the analysis of records in multiple languages.
The following feature extractor is implemented:

- agemagician/mlong-t5-tglobal-base [Source](https://huggingface.co/agemagician/mlong-t5-tglobal-base)

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
[`asreviewcontrib/models/mlong-t5-tglobal-base`](asreviewcontrib/models/mlong-t5-tglobal-base) and can be used in a simulation.

```bash
asreview simulate benchmark:van_de_Schoot_2017 -e mlongt5 -m svm
```

> Please note that, as with all transformers, this model produces negative vector values. Consequently, it is not compatible with Naive Bayes classifiers, which require non-negative feature values.


## License

MIT license
