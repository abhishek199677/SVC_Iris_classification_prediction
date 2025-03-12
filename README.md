# Iris_Classification_prediction

This project uses a supervised learning approach to classify Iris flowers into one of three species: Setosa, Versicolor, and Virginica. The model is trained on a dataset of 150 Iris flowers, each with 4 features. The goal is to predict the species of a new, unseen Iris flower based on its characteristics.

## 🔬 Features:

Sepal length (cm)
Sepal width (cm)
Petal length (cm)
Petal width (cm)


## 📌 Target Variable:

Species (Setosa, Versicolor, or Virginica)


## 🏗️ Model:

The project uses a logistic regression model to classify the Iris flowers. The model is trained on the training dataset and evaluated on the testing dataset.

## 🚀 Technologies Used:

1) Python
2) Scikit-learn
3) Streamlit
4) MongoDB

## Functionality:

The model takes in the characteristics of an Iris flower as input and predicts the species.
The model is trained on a dataset of 150 Iris flowers.
The model is evaluated on a testing dataset to measure its accuracy.
The model is deployed as a web application using Streamlit.
The model stores its predictions in a MongoDB database.

## 💍 Example Use Cases:

Predicting the species of an Iris flower based on its characteristics.
Identifying the characteristics of an Iris flower that are most important for predicting its species.
Comparing the accuracy of different machine learning models for Iris classification.

## 🎩 Getting Started:

## Clone the repository to your local machine.
conda create -n iris python=3.8 -y
conda activate iris
Install the required dependencies using pip install -r requirements.txt.
Run the model using streamlit run app.py.
Test the model by entering the characteristics of an Iris flower and predicting its species.
Contributing:

Contributions are welcome! To contribute to the project, please fork the repository and submit a pull request with your changes.

### License:

This project is licensed under the MIT License.


## 🏗️ Installation
### Prerequisites
Make sure you have the following installed:
- **Python 3.8+**
- **pip** (Python package manager)
- **MongoDB Atlas** (or a local MongoDB instance)

### Steps to Set Up
1. **Clone the repository:**
   ```sh
   git clone https://github.com/abhishek199677/Iris_Classification_prediction.git
   ```
2. **Create a virtual environment:**
   ```sh

   conda create -n iris python=3.8 -y              

    conda activate iris
   
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   pip freeze #must be done before running the app
   ```
4. **Set up Streamlit Secrets for MongoDB:**
   - Create a `.streamlit/secrets.toml` file:
   ```toml
   [mongodb]
   uri = "your_mongodb_connection_string"
   database = "your_database_name"
   collection = "your_collection_name"
   ```
5. **Run the app:**
   ```sh
   streamlit run app.py
   ```


## Final APP Details

https://irisclassificationpredictions.streamlit.app/
