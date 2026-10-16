# Spotify PCA and Factor Analysis

A statistical analysis project based on a dataset of 2,000 Spotify tracks. The project applies **Principal Component Analysis (PCA)** and **Factor Analysis** to explore relationships between audio features and identify the main dimensions that describe songs in the dataset.

## Project objective

The aim of the project is to reduce the dimensionality of the dataset and investigate whether several measured audio characteristics can be described through a smaller number of underlying components or factors.

The analysis focuses on the following variables:

- Beats Per Minute (BPM)
- Energy
- Danceability
- Loudness (dB)
- Liveness
- Valence
- Acousticness
- Speechiness

## Analysis workflow

The Python script performs the following steps:

1. Loads the Spotify dataset from `data_in/Spotify-2000.csv`.
2. Selects the numerical audio features used in the analysis.
3. Converts numeric values and handles missing observations using mean imputation.
4. Standardizes the variables with `StandardScaler`.
5. Builds the correlation matrix.
6. Performs Principal Component Analysis using the eigenvalues and eigenvectors of the correlation matrix.
7. Calculates component scores, communalities, cosines and contributions.
8. Generates PCA visualizations, including scree plots, score plots and correlation circles.
9. Checks whether factor analysis is appropriate using the Bartlett sphericity test and the KMO measure.
10. Performs factor analysis with and without Varimax rotation.
11. Exports the analysis tables and graphics to dedicated output folders.

## Main methods

### Principal Component Analysis

PCA is used to summarize the information contained in the original audio variables through a smaller set of uncorrelated components. The project calculates:

- eigenvalues;
- explained and cumulative variance;
- component scores;
- factor correlations;
- communalities;
- cosines;
- variable contributions.

### Factor Analysis

Factor analysis is used to identify latent dimensions behind the observed audio features. The project includes:

- Bartlett's test of sphericity;
- the Kaiser-Meyer-Olkin measure;
- factor loadings;
- communalities;
- factor scores;
- explained variance;
- Varimax rotation;
- comparison between rotated and unrotated solutions.

The number of factors is selected using the Kaiser criterion, based on eigenvalues greater than 1.

## Repository structure

```text
spotify-pca-factor-analysis/
├── data_in/
│   └── Spotify-2000.csv          # Input dataset
├── data_out/
│   ├── ACP/                      # PCA tables and component scores
│   └── Fact/                     # Factor analysis tables and scores
├── graphics/
│   ├── ACP/                      # PCA visualizations
│   └── Fact/                     # Factor analysis visualizations
├── main.py                       # Complete analysis pipeline
├── PROIECTDSADFM.docx            # Project report
└── README.md
```

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Factor Analyzer

## Installation

Install the required Python libraries with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn factor-analyzer
```

For a reproducible setup, the dependencies can also be placed in a `requirements.txt` file.

## Running the analysis

From the root directory of the repository, run:

```bash
python main.py
```

The script reads the input data from `data_in/Spotify-2000.csv` and automatically generates the analysis results in:

```text
data_out/ACP/
data_out/Fact/
graphics/ACP/
graphics/Fact/
```

## Output examples

The generated outputs include:

- correlation matrices;
- explained variance tables;
- component scores;
- PCA score plots;
- scree plots;
- correlation circles;
- PCA loading heatmaps;
- KMO and communalities heatmaps;
- rotated and unrotated factor loadings;
- rotated and unrotated factor scores.

## Notes

The project is designed as an academic statistical analysis workflow. The generated tables and graphics are kept in the repository to make the analysis process and results easier to inspect.
