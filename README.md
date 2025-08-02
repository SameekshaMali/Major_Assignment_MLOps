## MLOps Major Assisgnment - Linear Regression

This assignment demonstrates an MLOps pipeline that includes:
1. Data loading and preprocessing
2. Linear Regression model training
3. Evaluation using R² and MSE matrics
4. Manual 8-bit quantization of model parameters
5. Evaluation of the quantized model
6. CI/CD using GitHub actions
7. Containerization with Docker
8. Predict module to validate model inference inside container

├── src/
│ ├── train.py 
│ ├── quantize.py 
│ ├── predict.py 
│ ├── utils.py 
├── tests/
│ └── test_train.py 
├── models/ 
├── Dockerfile 
├── requirements.txt 
├── .github/workflows/
│ └── ci.yml 
└── README.md 

Dataset
1. California Housing Dataset from sklearn.datasets
2. Used to predict housing prices from 8 numerical characters

Setup Done -
1. Created Conda Environment.
2. Created train.py, quantize.py,predict.py

Model Performance Comparison
____________________________________________________________
Metric	   |     Original Model	   |     After Quantization |
____________________________________________________________
R² Score   |        0.5758	       |          0.5754        |
MSE	       |        0.5559	       |          0.5564        |
___________|_______________________|________________________|