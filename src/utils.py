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
    # Get all horizontal patterns.
    hor_array = get_all_horizontal_patterns(
        image_shape=image_shape,
        line_length=line_length,
        line_pixel_value=line_pixel_value,
    )

    # Create all vertical line patterns.
    ver_array = get_all_vertical_patterns(
        image_shape=image_shape,
        line_length=line_length,
        line_pixel_value=line_pixel_value,
    )

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
    """Get all horizontal patterns of the given image shape and length of the line as a flattened array.

    :param tuple[int, int] image_shape: image shape
    :param int line_length: length of line
    :param int line_pixel_value: value of line
    :return np.ndarray: all horizontal patterns as flattened
    :raise ValueError: if given line_length is larger than given image_shape[1]
    """
    if line_length > image_shape[1]:
        msg = f"""
            line_lenght must be equal to or less than image_shape[1],
            but line_length vs image_shape[1] = {line_length} vs {image_shape[1]}.
        """
        raise ValueError(msg)
    # Make the trivial pattern, which the line is set from the head.
    trivial_pattern = np.zeros(image_shape[1])
    trivial_pattern[:line_length] = line_pixel_value
    # Make the patterns for one line.
    num_patterns_for_one_line = image_shape[1] - (line_length - 1)
    patterns_for_one_line = np.zeros((num_patterns_for_one_line, image_shape[1]))
    for index in range(num_patterns_for_one_line):
        patterns_for_one_line[index, :] = np.roll(trivial_pattern, index)

    # Put the patterns for one line to every line.
    num_patterns = num_patterns_for_one_line * image_shape[0]
    image_length = image_shape[0] * image_shape[1]
    patterns = np.zeros((num_patterns, image_length))
    for index in range(image_shape[0]):
        start_row_index = index * num_patterns_for_one_line
        end_row_index = start_row_index + num_patterns_for_one_line
        start_column_index = index * image_shape[1]
        end_column_index = start_column_index + image_shape[1]
        patterns[start_row_index:end_row_index, start_column_index:end_column_index] = (
            patterns_for_one_line
        )

    return patterns


def get_all_vertical_patterns(
    image_shape: tuple[int, int], line_length: int, line_pixel_value: int
):
    image_length = image_shape[0] * image_shape[1]
    ver_shape = (4, image_length)
    ver_array = np.zeros(ver_shape)
    j = 0
    for i in range(0, 4):
        ver_array[j][i] = line_pixel_value
        ver_array[j][i + 4] = line_pixel_value
        j += 1

    return ver_array
