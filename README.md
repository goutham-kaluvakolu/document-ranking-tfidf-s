# Inaugural Address Information Retrieval System (IAIRS)

Welcome to the Inaugural Address Information Retrieval System (IAIRS)! This Python project offers a robust information retrieval system designed to analyze and extract insights from a curated corpus of 15 Inaugural addresses delivered by various US presidents.

## Requirements

- Python 3.5.1 or later.
- NLTK (Natural Language Toolkit) for text processing tasks.
- Standard Python libraries only, with the exception of NLTK.

## Dataset

The IAIRS operates on a carefully curated dataset comprising 15 .txt files, each containing the transcript of a presidential inaugural address.

## Functionalities

### Text Preprocessing

The IAIRS employs a comprehensive preprocessing pipeline, which encompasses the following steps:

- **Lowercasing**: Convert all text to lowercase for case-insensitive analysis.
- **Tokenization**: Employ NLTK's RegexpTokenizer to tokenize the text, splitting it into individual words or tokens.
- **Stopword Removal**: Utilize NLTK's built-in stopwords corpus to filter out common stopwords from the text.
- **Stemming**: Apply NLTK's Porter stemmer to reduce words to their base or root form, aiding in text normalization.

### TF-IDF Vectorization

The IAIRS implements the Term Frequency-Inverse Document Frequency (TF-IDF) algorithm to compute TF-IDF vectors for both documents and queries. It employs the ltc.lnc weighting scheme, which consists of the following components:

- **Document Weighting**: Utilize logarithmic TF, logarithmic IDF, and cosine normalization for document vectors.
- **Query Weighting**: Implement logarithmic TF, no IDF, and cosine normalization for query vectors.

### Query-Document Similarity Calculation

The IAIRS calculates the cosine similarity score between queries and documents based on their TF-IDF vectors. It retrieves the document with the highest similarity score as the query answer.

## Evaluation Criteria

The IAIRS will be evaluated based on the following criteria:

- Correctness and accuracy of implemented functions (getidf, getweight, query).
- Efficiency and effectiveness of preprocessing steps.
- Modularity, maintainability, and adherence to coding standards.

## Usage

1. Ensure Python and NLTK are installed.
2. Download NLTK data using `python -m nltk.downloader all`.
3. Clone the repository and navigate to the project directory.
4. Place the dataset files in the designated directory.
5. Execute the Python script (`presidential_search.py`).

Example usage:

```bash
python presidential_search.py

IAIRS/
│
├── README.md
├── presidential_search.py
└── dataset/
    ├── 01_washington_1789.txt
    ├── 02_washington_1793.txt
    ...
    └── 15_obama_2013.txt

