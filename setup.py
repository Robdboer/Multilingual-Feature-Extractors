from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name='asreview-multilingual-feature-extractors-extension',
    version='1.0',
    description='Multilingual feature extractors extension',
    url='https://github.com/robdboer/multilingual-sentence-transformers',
    author='Rob den Boer',
    author_email='robbiedboer99@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='systematic review',
    packages=find_namespace_packages(include=['asreviewcontrib.*']),
    python_requires='~=3.6',
    install_requires=[
        'asreview>=1.0'
    ],
    entry_points={
        'asreview.models.classifiers': [
            # define classifier algorithms
        ],
        'asreview.models.feature_extraction': [
            # define feature_extraction algorithms
            "multilingual = asreviewcontrib.models.paraphrase_multilingual_MiniLM_L12_v2:paraphrasemultilingualMiniLML12v2",
        ],
        'asreview.models.balance': [
            # define balance strategy algorithms
        ],
        'asreview.models.query': [
            # define query strategy algorithms
        ]
    },
    project_urls={
        'Bug Reports': 'https://github.com/asreview/asreview/issues',
        'Source': 'https://github.com/robdboer/multilingual-sentence-transformers/',
    },
)
