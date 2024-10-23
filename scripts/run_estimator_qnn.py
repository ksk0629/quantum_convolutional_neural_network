import argparse
import os
import yaml

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
    config_general = config["general"]
    config_dataset = config["dataset"]
    config_model = config["model"]
    config_train = config["train"]

    # Fix the random seed.
    fix_seed(config_general["random_seed"])

    # Create the dataset.
    images, labels = generate_line_dataset(**config_dataset["generating"])
    print("Generated the dataset.")

    # Get the training and testing datasets.
    train_images, test_images, train_labels, test_labels = train_test_split(
        images,
        labels,
        test_size=config_dataset["settings"]["test_size"],
        random_state=config_general["random_seed"],
    )

    # Get the QNN.
    match config_model["mode"]:
        case "example_estimator":
            qnn = QNNBuilder().get_example_structure_estimator_qnn(len(train_images[0]))
        case "exact_aer_estimator":
            qnn = QNNBuilder().get_exact_aer_estimator_qnn(len(train_images[0]))
        case "noisy_aer_estimator":
            qnn = QNNBuilder().get_noisy_aer_estimator_qnn(len(train_images[0]))
        case "example_sampler":
            qnn = QNNBuilder().get_example_structure_sampler_qnn(len(train_images[0]))
        case "exact_aer_sampler":
            qnn = QNNBuilder().get_exact_aer_sampler_qnn(len(train_images[0]))
        case "noisy_aer_sampler":
            qnn = QNNBuilder().get_noisy_aer_sampler_qnn(len(train_images[0]))
    print(f"Built the QNN, given mode: {config_model["mode"]}.")

    # Create the classifier.
    qnn_trainer = QNNTrainer(
        qnn=qnn,
        train_data=train_images,
        train_labels=train_labels,
        test_data=test_images,
        test_labels=test_labels,
        callback=callback_print,
        seed=None,  # Already fixed.
    )
    print("Built the QNNTrainer.")

    # Create the directory to save the model.
    model_path = config_train["model_path"]
    dir_path = os.path.dirname(model_path)
    if os.path.isdir(dir_path):
        os.makedirs(dir_path)
    # Fit the model.
    match config_train["optimiser"]:
        # --- Local optimisers ---
        case "adam":
            optimiser = qiskit_algorithms.optimizers.ADAM
        case "adgd":
            optimiser = qiskit_algorithms.optimizers.ADGS
        case "cg":
            optimiser = qiskit_algorithms.optimizers.CG
        case "cobyla":
            optimiser = qiskit_algorithms.optimizers.COBYLA
        case "l_bfgs_b":
            optimiser = qiskit_algorithms.optimizers.L_BFGS_B
        case "gsls":
            optimiser = qiskit_algorithms.optimizers.GSLS
        case "gradient_descent":
            optimiser = qiskit_algorithms.optimizers.GradientDescent
        case "gradient_descent_state":
            optimiser = qiskit_algorithms.optimizers.GradientDescentState
        case "nelder_mead":
            optimiser = qiskit_algorithms.optimizers.NELDER_MEAD
        case "nft":
            optimiser = qiskit_algorithms.optimizers.NFT
        case "p_bfgs":
            optimiser = qiskit_algorithms.optimizers.P_BFGS
        case "powell":
            optimiser = qiskit_algorithms.optimizers.POWELL
        case "slsqp":
            optimiser = qiskit_algorithms.optimizers.COBYLA
        case "spsa":
            optimiser = qiskit_algorithms.optimizers.SPSA
        case "qnspsa":
            optimiser = qiskit_algorithms.optimizers.QNSPSA
        case "tnc":
            optimiser = qiskit_algorithms.optimizers.TNC
        case "scipy_optimiser":
            optimiser = qiskit_algorithms.optimizers.SciPyOptimiser
        case "umda":
            optimiser = qiskit_algorithms.optimizers.UMDA
        case "bobyqa":
            optimiser = qiskit_algorithms.optimizers.BOBYQN
        case "imfil":
            optimiser = qiskit_algorithms.optimizers.IMFIL
        case "snobfit":
            optimiser = qiskit_algorithms.optimizers.SNOBFIT
        # --- global optimisers ---
        case "crs":
            optimiser = qiskit_algorithms.optimizers.CRS
        case "direct_l":
            optimiser = qiskit_algorithms.optimizers.DIRECT_L
        case "direct_l_rand":
            optimiser = qiskit_algorithms.optimizers.DIRECT_L_RAND
        case "esch":
            optimiser = qiskit_algorithms.optimizers.ESCH
        case "isres":
            optimiser = qiskit_algorithms.ISRES
        case _:
            optimiser = None
    print(
        f"Get optimiser, given optimiser: {config_train["optimiser"]}, the instance: {optimiser}."
    )

    qnn_trainer.fit(
        model_path=model_path,
        optimiser=optimiser,
        loss=config_train["loss"],
        optimiser_settings=config_train["optimiser_settings"],
    )
