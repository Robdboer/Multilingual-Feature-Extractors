# ASReview feature matrix reader

This extension to ASReview implements a feature extraction model that reads existing feature matrices.

## Getting started

Install the multilingual feature extractors with:

```bash
pip install git+https://github.com/Robdboer/multilingual-feature-extractors.git
```

## Usage

The feature matrix reader is defined in
[`asreviewcontrib/models/readfm`](asreviewcontrib/models/readfm) and can be used in a simulation.

```bash
asreview simulate benchmark:van_de_Schoot_2017 -e readfm -m svm
```

> Please note that that this feature matrix reader requires a numpy feature matrix file in the simulation folder.


## License

MIT license
