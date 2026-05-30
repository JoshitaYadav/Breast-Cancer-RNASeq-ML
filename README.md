# Breast-Cancer-RNASeq-ML
Breast Cancer RNA-Seq Biomarker Identification and Cancer Classification Using Machine Learning and Random Forest.
# Breast Cancer RNA-Seq Biomarker Identification and Cancer Classification Using Machine Learning

## Overview

This repository contains the source code, datasets, methodology, and results associated with the B.Tech Biotechnology final-year project:

**Breast Cancer RNA-Seq Biomarker Identification and Cancer Classification Using Machine Learning: A Random Forest-Based Transcriptomic Analysis**

The study investigates the use of RNA sequencing (RNA-seq) gene expression data and machine learning techniques for identifying breast cancer-associated transcriptomic biomarkers and classifying cancerous and non-cancerous breast tissue samples.

---

## Dataset

Gene Expression Omnibus (GEO)

Accession:

GSE183947

Dataset Link:

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183947

Samples:

* 30 Breast Cancer Samples
* 30 Matched Normal Samples

Total Samples:

60

Genes:

20,246

---

## Methodology

1. RNA-seq data preprocessing
2. Log1p transformation
3. Feature selection using ANOVA F-test
4. Random Forest Classification
5. Stratified Cross Validation
6. Stratified Group Cross Validation
7. Permutation Testing
8. Feature Importance Analysis

---

## Model Performance

### Holdout Test Set

Accuracy: 100%

Recall: 100%

Precision: 100%

ROC-AUC: 1.000

### Stratified 5-Fold Cross Validation

Accuracy:

98.33% ± 3.73%

### Permutation Test

Accuracy after label randomization:

48.5%

This confirms that the model is learning biological signal rather than random patterns.

---

## Repository Structure

scripts/ : Source code

models/ : Saved machine learning models

results/ : Generated figures and metrics

dissertation/ : Final report

docs/ : Workflow diagrams and supporting figures

---

## Requirements

Python 3.10+

scikit-learn

pandas

numpy

matplotlib

seaborn

joblib

---

## Citation

If you use this repository, please cite:

Yadav, J. (2026). Breast Cancer RNA-Seq Biomarker Identification and Cancer Classification Using Machine Learning. GitHub Repository.
