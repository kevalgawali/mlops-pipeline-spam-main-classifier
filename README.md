# Spam Email Classification with End-to-End MLOps Pipeline

## Project Overview
This project focuses on building an **end-to-end pipeline** for a spam email classifier using Machine Learning.
Along with model training, it implements MLOps practices like data versioning,
experiment tracking, and reproducibility using DVC and AWS S3.

## Features
- Spam vs Ham email classification
- End-to-end ML pipeline
- Data and model versioning using **DVC**
- Experiment tracking using **DVCLive**
- **AWS S3** Remote storage

## Tech Stack
- Programming Language: **Python**
- Machine Learning: **Scikit-learn**
- Data cleaning and Manipulation: **Pandas**
- MLOps Tool: **DVC** for data versioning
- Version Control: **Git**
- Cloud Storage: **AWS S3**

## MLOps Workflow
1. Data ingestion
2. Data preprocessing
3. Feature Engineering
4. Model training
5. Model Evaluation
6. Data versioning using DVC
7. Experiment tracking
8. Model artifacts stored in AWS S3
9. Reproducible pipeline using `dvc repro`

## Project Structure
```
├── data/
├── logs/
├── models/
├── reports/
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   ├── model_evaluation.py
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── README.md
```


## Conclusion
This project demonstrates how Machine Learning models can be developed and
managed using real-world MLOps practices.

By integrating **DVC** for data and experiment versioning, **Git** for code management,
and **AWS S3** for remote storage, the project ensures reproducibility, scalability,
and structured experimentation beyond notebook-based workflows.
