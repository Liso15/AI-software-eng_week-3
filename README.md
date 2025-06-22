# AI-software-eng_week-3

├── task1_classical_ml/
│   ├── iris_classification.py
│   ├── iris_decision_tree.png
│   └── results.png
├── task2_deep_learning/
│   ├── mnist_cnn.py
│   ├── model_architecture.png
│   ├── training_metrics.png
│   └── sample_predictions.png
├── task3_nlp/
│   ├── amazon_reviews_analysis.py
│   ├── ner_visualization.png
│   └── sentiment_analysis.png
├── requirements.txt
└── README.md
## Task 1: Classical ML with Scikit-learn (Iris Classification)

Overview
- Dataset: Iris Species (150 samples, 4 features, 3 classes)

- Model: Decision Tree Classifier

- Goal: Predict iris species with high accuracy

## Results
- Metric	Value
- Accuracy	96.7%
- Precision	96.8%
- Recall	96.7%
## Screenshorts
![scatter_plot](https://github.com/user-attachments/assets/1b02de31-4092-41a5-8bdb-18e2c18bd516)
![confusion_matrix](https://github.com/user-attachments/assets/26c4a060-99c7-4ca4-b56c-e401c62be8b8)
![correlation_matrix](https://github.com/user-attachments/assets/6019951f-b281-4f92-aba1-2b8940affbd6)
![feature_distributions](https://github.com/user-attachments/assets/2001c587-fee2-49d6-895f-81c449808136)
![feature_importance](https://github.com/user-attachments/assets/6af02877-e09b-4025-a1e5-babbdd37dbc4)
![scatter_plot](https://github.com/user-attachments/assets/6cc37249-ca61-43b6-959e-62287527d074)

## Task 2: Deep Learning with TensorFlow (MNIST Digit Classification)

- Overview
- Dataset: MNIST Handwritten Digits (70,000 28x28 grayscale images)
- Model: Convolutional Neural Network (CNN)
- Goal: Achieve >95% test accuracy
  
 ## Training Results
- Test Accuracy: 98.3%
- Training Time: 5 minutes (10 epochs on CPU)

## Screenshorts
![Figure_1](https://github.com/user-attachments/assets/2e3c1dfd-08d8-400d-b06f-49d26777d666)
![figure_2](https://github.com/user-attachments/assets/c623e6d9-bd54-4865-97b3-84baade9c915)
![Figure_3](https://github.com/user-attachments/assets/6b060bcf-7cfb-4f1b-99cf-130b166987bc)

## Task 3: NLP with spaCy (Amazon Review Analysis)

- Overview
- Dataset: Amazon Product Reviews
- Tasks:
- Named Entity Recognition (product names, brands)
- Rule-based sentiment analysis

## Results 
Entities:
- iPhone (PRODUCT)
- Apple (ORG)

Sentiment: Positive (Positive words: 'revolutionary').

## Setup Instructions

git clone https://github.com/yourusername/ml-portfolio.git
cd ml-portfolio

git clone https://github.com/yourusername/ml-portfolio.git
cd ml-portfolio

python -m spacy download en_core_web_sm

# Task 1
python task1_classical_ml/iris_classification.py

# Task 2
python task2_deep_learning/mnist_cnn.py

# Task 3
python task3_nlp/amazon_reviews_analysis.py

## Requirements
Python 3.8+

scikit-learn

TensorFlow 2.x

spaCy

matplotlib

pandas

seaborn 

## Collobrator 
- Liso Mlunguza
- Email: lisomlunguza8@gmail.com
  








