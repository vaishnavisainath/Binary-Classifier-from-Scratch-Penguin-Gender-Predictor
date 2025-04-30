# 🐧 Binary Classifier from Scratch - Penguin Gender Predictor

This project implements a binary classifier to predict the gender of penguins using a custom logistic regression model. It includes data preprocessing, feature engineering, model training, and evaluation.

## 📂 Dataset

The dataset used is `penguins.csv`, which contains information about penguins' physical characteristics. The main features are:

- `bill_length_mm`
- `bill_depth_mm`
- `flipper_length_mm`
- `body_mass_g`
- `sex` (Target variable)

## 🛠️ Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Pickle (for saving model)

## 🔍 Project Highlights

- Cleaned the dataset by handling missing values
- Encoded categorical target variable (`sex`)
- Normalized numerical features
- Implemented logistic regression from scratch
- Evaluated the model's performance by calculating accuracy
- Saved the trained model using Pickle

## 📈 Output

The model predicts whether a penguin is male or female based on its physical attributes. The accuracy of the model is calculated and output after training and testing the logistic regression model.
