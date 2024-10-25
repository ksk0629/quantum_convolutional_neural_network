import argparse
import yaml

from sklearn.model_selection import train_test_split

from src.training import train
from src.utils import fix_seed, generate_line_dataset

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

    train(
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
        seed=None,  # Already fixed
    )
