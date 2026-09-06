# CIC-IoT2023 Dataset Study

## 1. Dataset Purpose

CIC-IoT2023 is used as the second network intrusion detection
dataset for the Hybrid ML-Based Intrusion Detection System.

It provides network traffic generated in an IoT environment and
contains benign and malicious traffic.

## 2. Dataset Organization

The available data is provided as multiple CSV files.

Examples include attack-specific files such as:

- DDoS-PSHACK_FLOOD.csv
- Other DDoS attack files
- Other attack-category CSV files

## 3. Dataset Structure

Each CSV file contains network traffic records represented by
multiple network-flow features.

The exact number of rows and columns for each file will be recorded
during the dataset inspection notebook.

## 4. Dataset Organization Difference

Unlike the CICIDS2017 collection, where CSV files are associated
with different capture periods and attack scenarios, the available
CIC-IoT2023 CSV collection is organized around individual traffic
or attack categories.

This difference will be considered when designing the common
dataset-loading pipeline.

## 5. Planned Usage

CIC-IoT2023 will be used for:

1. Dataset analysis
2. Data cleaning
3. Feature analysis
4. Exploratory data analysis
5. Model development
6. Model evaluation
7. Hybrid IDS validation

## 6. Initial Investigation

The following properties will be investigated for every CSV:

- Number of records
- Number of features
- Column names
- Data types
- Missing values
- Duplicate records
- Label/target representation
- Class distribution
- Infinite values

## 7. Planned Processing

The raw CIC-IoT2023 files will remain unchanged.

The processing pipeline will be:

Raw CSV
→ Loading
→ Cleaning
→ Feature/label identification
→ Encoding
→ Scaling
→ Train/Validation/Test split
→ Model training

Detailed preprocessing will be implemented in later weeks.