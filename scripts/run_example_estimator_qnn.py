import numpy as np
import qiskit_algorithms
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split

from src.qnn_builder import QNNBuilder
from src.utils import generate_line_dataset

if __name__ == "__main__":
    # Get the dataset.
    images, labels = generate_line_dataset(50)

    # Get the training and testing datasets.
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=91
    )

    # Get the qiskit example QNN.
    example_estimator_qnn = QNNBuilder(len(train_images[0])).get_example_estimator_qnn()

    # Create the classifier.
    classifier = NeuralNetworkClassifier(
        example_estimator_qnn,
        optimizer=qiskit_algorithms.optimizers.COBYLA(maxiter=200),
    )
    # Fit the model.
    x = np.asarray(train_images)
    y = np.asarray(train_labels)
    classifier.fit(x, y)
    print(
        f"Accuracy from the train data : {np.round(100 * classifier.score(x, y), 2)}%"
    )

    # Test the model.
    x = np.asarray(test_images)
    y = np.asarray(test_labels)
    print(f"Accuracy from the test data: {np.round(100 * classifier.score(x, y), 2)}%")

    # Save the model.
    model_path = "models/example_qnn.model"
    classifier.save(model_path)

    # Load the saved model as the test.
    save_classifier = NeuralNetworkClassifier.load(model_path)

    # Test the saved model as the test.
    print(
        f"Accuracy from the test data: {np.round(100 * save_classifier.score(x, y), 2)}%"
    )
