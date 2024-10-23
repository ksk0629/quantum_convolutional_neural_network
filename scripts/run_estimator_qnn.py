import argparse
import os
import yaml

import numpy as np
import qiskit_algorithms
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from sklearn.model_selection import train_test_split

from src.qnn_builder import QNNBuilder
from src.utils import fix_seed, generate_line_dataset, callback_print

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and Evaluate the QCNN with a given line dataset."
    )
    parser.add_argument("-c", "--config_yaml_path", required=False, type=str)
    args = parser.parse_args()

    # Read the given config file.
    with open(args.config_yaml_path, "r") as config_yaml:
        config = yaml.safe_load(config_yaml)
    config_train = config["train"]
    config_dataset = config["dataset"]

    # Fix the random seed.
    fix_seed(config["train"]["random_seed"])

    # Create the dataset.
    print("Generating the dataset...", end="")
    images, labels = generate_line_dataset(**config["dataset"])
    print("Done.")

    # Get the training and testing datasets.
    train_images, test_images, train_labels, test_labels = train_test_split(
        images,
        labels,
        test_size=config_train["test_size"],
        random_state=config_train["random_seed"],
    )

    # Get the qiskit example QNN.
    print("Building the model...", end="")
    example_estimator_qnn = QNNBuilder().get_example_structure_estimator_qnn(
        len(train_images[0])
    )
    print("Done.")

    # Create the classifier.
    classifier = NeuralNetworkClassifier(
        example_estimator_qnn,
        optimizer=qiskit_algorithms.optimizers.COBYLA(maxiter=config_train["maxiter"]),
        callback=callback_print,
    )
    # Fit the model.
    print("Fitting the model...", end="")
    x = np.asarray(train_images)
    y = np.asarray(train_labels)
    classifier.fit(x, y)
    print("Done.")
    print(
        f"Accuracy from the train data : {np.round(100 * classifier.score(x, y), 2)}%"
    )

    # Test the model.
    x = np.asarray(test_images)
    y = np.asarray(test_labels)
    print(f"Accuracy from the test data: {np.round(100 * classifier.score(x, y), 2)}%")

    model_path = config_train["model_path"]
    # Create the directory.
    dir_path = os.path.dirname(model_path)
    if os.path.isdir(dir_path):
        os.makedirs(dir_path)
    # Save the model.
    classifier.save(model_path)

    # Load the saved model as the test.
    save_classifier = NeuralNetworkClassifier.load(model_path)

    # Test the saved model as the test.
    print(
        f"Accuracy from the test data: {np.round(100 * save_classifier.score(x, y), 2)}%"
    )
