import numpy as np
import qiskit_algorithms
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from sklearn.model_selection import train_test_split

from src.qnn_builder import QNNBuilder
from src.utils import fix_seed, generate_line_dataset, callback_print

if __name__ == "__main__":
    # Fix the random seed.
    seed = 91
    fix_seed(seed)

    # Get the dataset.
    images, labels = generate_line_dataset(50)

    # Get the training and testing datasets.
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=seed
    )

    # Get the qiskit example QNN.
    example_sampler_qnn = QNNBuilder().get_example_noisy_aer_sampler_qnn(seed=seed)

    # Create the classifier.
    classifier = NeuralNetworkClassifier(
        example_sampler_qnn,
        optimizer=qiskit_algorithms.optimizers.COBYLA(maxiter=200),
        callback=callback_print,
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
    model_path = "models/example_qnn_with_noisy_aer.model"
    classifier.save(model_path)

    # Load the saved model as the test.
    save_classifier = NeuralNetworkClassifier.load(model_path)

    # Test the saved model as the test.
    print(
        f"Accuracy from the test data: {np.round(100 * save_classifier.score(x, y), 2)}%"
    )
