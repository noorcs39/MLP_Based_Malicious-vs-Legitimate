# MLP Model for Malicious vs. Legitimate Classification

This project demonstrates the use of a Multilayer Perceptron (MLP) model to classify user behavior as either "Malicious" or "Legitimate" based on energy consumption and user data. The MLP is trained on a dataset containing features related to energy usage and user identifiers, making it a powerful tool for detecting potential security threats.

## Project Structure

- **Dataset.xlsx**: The dataset containing energy consumption data and user behavior labels (Malicious vs. Legitimate).
- **Figure_1.png**: Graph showing the training and validation loss over epochs.
- **Figure_2.png**: Graph showing the training, validation accuracy, and testing accuracy.
- **run_MLP.py**: The main script that runs the MLP model, including data preprocessing, model training, evaluation, and visualization.

## Installation

To run this project, you need to have Python installed along with the following packages:

- `tensorflow`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `openpyxl` (for reading Excel files)

## You can install the necessary packages using pip:
pip install tensorflow pandas scikit-learn matplotlib seaborn openpyxl

## Output

The script will generate the following:

- A classification report showing precision, recall, F1-score, and support for both classes.
- A confusion matrix plot visualizing the model's performance.
- Two figures (`Figure_1.png` and `Figure_2.png`) illustrating the loss and accuracy metrics over the training process.

## Example Output

- **Figure_1.png**: Displays the training and validation loss across all epochs, helping to identify overfitting or underfitting.
- **Figure_2.png**: Shows the training and validation accuracy trends, with a horizontal line representing the testing accuracy.

## Project Overview

The project is designed to tackle a binary classification problem, where the goal is to distinguish between legitimate and malicious user behavior based on specific features. The MLP model is trained using TensorFlow and evaluated using standard metrics such as accuracy, precision, recall, and F1-score.

## Key Features

- **Binary Classification**: Classify user behavior as "Malicious" or "Legitimate".
- **Data Visualization**: Graphical representation of training progress and model evaluation.
- **Confusion Matrix**: Visualize the performance of the classification model.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute this code as needed.

## Author

Nooruddin Noonari
