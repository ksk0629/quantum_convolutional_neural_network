import numpy as np
import qiskit_algorithms


def generate_line_dataset(
    num_images: int,
    image_shape: tuple[int, int] = (2, 4),
    line_length: int = 2,
    line_pixel_value: float = np.pi / 2,
    min_noise_value: float = 0,
    max_noise_value: float = np.pi / 4,
):
    # Define the hyper-params.
    image_length = image_shape[0] * image_shape[1]  # The images are flattened to store.

    # Get all horizontal patterns.
    hor_array = get_all_horizontal_patterns(
        image_shape=image_shape,
        line_length=line_length,
        line_pixel_value=line_pixel_value,
    )

    # Create all vertical line patterns.
    ver_shape = (4, image_length)
    ver_array = np.zeros(ver_shape)
    j = 0
    for i in range(0, 4):
        ver_array[j][i] = line_pixel_value
        ver_array[j][i + 4] = line_pixel_value
        j += 1

    # Generate random images.
    images = []
    labels = []
    for _ in range(num_images):
        rng = qiskit_algorithms.utils.algorithm_globals.random.integers(0, 2)
        if rng == 0:
            labels.append(-1)
            random_image = qiskit_algorithms.utils.algorithm_globals.random.integers(
                0, 6
            )
            images.append(np.array(hor_array[random_image]))
        elif rng == 1:
            labels.append(1)
            random_image = qiskit_algorithms.utils.algorithm_globals.random.integers(
                0, 4
            )
            images.append(np.array(ver_array[random_image]))

        # Create noise.
        for i in range(8):
            if images[-1][i] == 0:
                images[-1][i] = (
                    qiskit_algorithms.utils.algorithm_globals.random.uniform(
                        min_noise_value, max_noise_value
                    )
                )

    return images, labels


def get_all_horizontal_patterns(
    image_shape: tuple[int, int], line_length: int, line_pixel_value: int
) -> np.ndarray:
    """Get all horizontal patterns of the given image shape and length of the line.

    :param tuple[int, int] image_shape: image shape
    :param int line_length: length of line
    :param int line_pixel_value: value of line
    :return np.ndarray: flattened all horizontal patterns
    """
    image_length = image_shape[0] * image_shape[1]
    hor_shape = (6, image_length)
    hor_array = np.zeros(hor_shape)
    j = 0
    for i in range(0, 7):
        if i != 3:
            hor_array[j][i] = line_pixel_value
            hor_array[j][i + 1] = line_pixel_value
            j += 1
    return hor_array
