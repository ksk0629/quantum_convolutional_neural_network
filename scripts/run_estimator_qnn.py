import argparse
import os
import yaml

import numpy as np
import qiskit_algorithms
from sklearn.model_selection import train_test_split

from src.qnn_builder import QNNBuilder
from src.qnn_trainer import QNNTrainer
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
    qnn_trainer = QNNTrainer(
        qnn=example_estimator_qnn,
        train_data=train_images,
        train_labels=train_labels,
        test_data=test_images,
        test_labels=test_labels,
        callback=callback_print,
        seed=None,  # Already fixed.
    )

    # Create the directory to save the model.
    model_path = config_train["model_path"]
    dir_path = os.path.dirname(model_path)
    if os.path.isdir(dir_path):
        os.makedirs(dir_path)
    # Fit the model.
    qnn_trainer.fit(
        model_path=model_path,
        optimiser=qiskit_algorithms.optimizers.COBYLA,
        loss="squared_error",
        optimiser_settings={"maxiter": config_train["maxiter"]},
    )
