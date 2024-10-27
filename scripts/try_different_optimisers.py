import numpy as np
from sklearn.model_selection import train_test_split
from qiskit.primitives import Sampler
import qiskit_algorithms

from src.line_dataset_generator import LineDatasetGenerator
from src.qnn_builder import QNNBuilder
from src.training import train
from src.utils import fix_seed

ERROR_TEST_OPTIMISERS = ["hogehoge"]
LOCAL_OPTIMISERS = [
    "adam",
    "adgd",
    "cg",
    "cobyla",
    "l_bfgs_b",
    "gsls",
    "gradient_descent",
    "nelder_mead",
    "nft",
    "powell",
    "slsqp",
    "spsa",
    "tnc",
    "umda",
    "bobyqn",
    "imfil",
    "snobfit",
]
LOCAL_OPTIMISERS_WITH_FIDELITY = ["qnspsa"]
GLOBAL_OPTIMISERS = ["crs", "direct_l", "direct_l_rand", "esch", "isres"]

ERROR_TEXT_PATH = "./error_from_try_different_optimisers.txt"
EXPERIMENT_NAME = "try_different_optimisers"

if __name__ == "__main__":
    # Define the settings.
    config_general = {"random_seed": 91}
    config_dataset = {
        "generating": {
            "num_images": 50,
            "image_shape": [4, 4],
            "line_length": 2,
            "line_pixel_value": np.pi / 2,
            "min_noise_value": 0,
            "max_noise_value": np.pi / 4,
        },
        "settings": {"test_size": 0.3},
    }
    config_model = {"mode": "example_estimator"}
    config_train = {
        "callback": "callback_mlflow",
        "loss": "squared_error",
        "optimiser_settings": {},
    }
    config_mlflow = {"experiment_name": "issue_11_different_optimisers"}

    # Fix the random seed.
    fix_seed(config_general["random_seed"])

    # Create the dataset.
    num_images = 50
    del config_dataset["generating"]["num_images"]
    line_dataset = LineDatasetGenerator(**config_dataset["generating"])
    images, labels = line_dataset.generate(num_images=num_images)

    # Get the training and testing datasets.
    train_images, test_images, train_labels, test_labels = train_test_split(
        images,
        labels,
        test_size=config_dataset["settings"]["test_size"],
        random_state=config_general["random_seed"],
    )

    for optimisers in [LOCAL_OPTIMISERS, GLOBAL_OPTIMISERS]:
        for optimiser in optimisers:
            try:
                run_name = f"{optimiser}_default"
                config_train["model_path"] = f"./models/{EXPERIMENT_NAME}/{run_name}"
                config_train["optimiser"] = optimiser
                train(
                    experiment_name=EXPERIMENT_NAME,
                    run_name=run_name,
                    train_data=train_images,
                    train_labels=train_labels,
                    test_data=test_images,
                    test_labels=test_labels,
                    mode=config_model["mode"],
                    model_path=config_train["model_path"],
                    optimiser_str=config_train["optimiser"],
                    loss=config_train["loss"],
                    initial_point=None,
                    callback_str=config_train["callback"],
                    optimiser_settings=config_train["optimiser_settings"],
                    seed=config_general["random_seed"],
                )
            except Exception as e:
                with open(ERROR_TEXT_PATH, "a") as error_file:
                    print(f"Error happened during {optimiser}.")
                    error_file.write(f"{optimiser}: {repr(e)}\n")

    # This one needs non-default argument, so run it separately.
    try:
        sampler = Sampler()
        ansatz = QNNBuilder()._get_ansatz(data_size=len(train_images[0]))
        fidelity = qiskit_algorithms.optimizers.QNSPSA.get_fidelity(ansatz, sampler)
        optimiser = "anspsa"
        run_name = f"{optimiser}_default"
        config_train["model_path"] = f"./models/{EXPERIMENT_NAME}/run_name"
        config_train["optimiser"] = optimiser
        train(
            experiment_name=EXPERIMENT_NAME,
            run_name=run_name,
            train_data=train_images,
            train_labels=train_labels,
            test_data=test_images,
            test_labels=test_labels,
            mode=config_model["mode"],
            model_path=config_train["model_path"],
            optimiser_str=config_train["optimiser"],
            loss=config_train["loss"],
            initial_point=None,
            callback_str=config_train["callback"],
            optimiser_settings=config_train["optimiser_settings"],
            seed=config_general["random_seed"],
        )
    except Exception as e:
        with open(ERROR_TEXT_PATH, "a") as error_file:
            print(f"Error happened during {optimiser}.")
            error_file.write(f"{optimiser}: {repr(e)}\n")
