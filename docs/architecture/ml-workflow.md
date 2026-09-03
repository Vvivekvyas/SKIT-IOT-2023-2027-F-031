\# ML Workflow Architecture



\## Hybrid ML-Based Intrusion Detection System



\### 1. Data Sources



The system uses the following network intrusion detection datasets:



\- CICIDS2017

\- CIC-IoT2023



These datasets provide network traffic features and attack labels for training and evaluating the intrusion detection models.



\### 2. Data Loading



The datasets are loaded into the ML pipeline using reusable data-loading utilities.



\*\*Input:\*\* Raw CSV dataset files  

\*\*Output:\*\* Loaded dataset for preprocessing



\### 3. Data Cleaning



The loaded data is cleaned before model training.



Main operations include:



\- Removing unnecessary spaces from column names

\- Handling missing values

\- Handling duplicate records

\- Checking invalid and infinite values

\- Validating feature consistency



\*\*Input:\*\* Raw loaded data  

\*\*Output:\*\* Clean dataset



\### 4. Encoding and Scaling



The cleaned data is transformed into a format suitable for machine learning.



Operations include:



\- Label encoding

\- Feature encoding where required

\- Numerical feature scaling

\- Preparing model-compatible feature matrices



\*\*Input:\*\* Clean dataset  

\*\*Output:\*\* Preprocessed dataset



\### 5. Exploratory Data Analysis



EDA is performed to understand the dataset and attack distribution.



Analysis includes:



\- Class distribution

\- Attack-type frequency

\- Feature distributions

\- Dataset statistics

\- Identification of class imbalance



\*\*Output:\*\* EDA results and visualizations



\### 6. Train / Validation / Test Split



The preprocessed dataset is divided into:



\- Training set

\- Validation set

\- Testing set



The split is designed to provide reliable model training and evaluation.



\### 7. ML Models



The project evaluates multiple machine learning approaches.



\#### Baseline Models

Initial ML models are trained to establish baseline performance.



\#### FT-Transformer

A tabular-data transformer model is implemented for supervised attack classification.



\#### Autoencoder

An Autoencoder is implemented for anomaly detection using reconstruction-based analysis.



\#### Hybrid Model

The FT-Transformer and Autoencoder are combined to create the final hybrid intrusion detection approach.



\### 8. Model Evaluation



The models are evaluated using:



\- Accuracy

\- Precision

\- Recall

\- F1-score



The performance of the individual models is compared before selecting the final hybrid model.



\### 9. Final Hybrid Model



The selected FT-Transformer + Autoencoder configuration becomes the final ML model used by the IDS.



\*\*Input:\*\* Preprocessed network traffic features  

\*\*Output:\*\* Attack prediction and anomaly information



\### 10. ML Inference



The trained model is packaged for inference.



The inference pipeline performs:



1\. Input validation

2\. Preprocessing

3\. Model prediction

4\. Attack classification

5\. Confidence/anomaly score generation



\### 11. Backend / API Integration



The ML inference component is integrated with the backend.



The backend receives network traffic features, passes them to the ML inference pipeline, and receives the model prediction.



\### 12. Prediction and Alert



The final system produces an intrusion detection result.



Example:



```json

{

&#x20; "prediction": "DDoS",

&#x20; "confidence": 0.96,

&#x20; "anomaly\_score": 0.12

}

