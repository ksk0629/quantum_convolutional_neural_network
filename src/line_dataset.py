import numpy as np

import qiskit_algorithms


class LineDataset:
    """Line dataset class.
    This class is generalisation of the dataset introduced in the qiskit tutorial.
    See https://qiskit-community.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html.
    """

    def __init__(
        self,
        image_shape: tuple[int, int] = (2, 4),
        line_length: int = 2,
        line_pixel_value: float = np.pi / 2,
        min_noise_value: float = 0,
        max_noise_value: float = np.pi / 4,
    ):
        """Initialise the class.

        :param tuple[int, int] image_shape: image shape, defaults to (2, 4)
        :param int line_length: length of line, defaults to 2
        :param float line_pixel_value: value of line, defaults to np.pi/2
        :param float min_noise_value: minimum value for noise, defaults to 0
        :param float max_noise_value: maximum value for noise, defaults to np.pi/4
        :raises ValueError: if given line_length is larger than given image_shape[0] or image_shape[1]
        """
        # Check if the given line_length is not greater than both image_shape.
        if line_length > image_shape[0] or line_length > image_shape[1]:
            msg = f"""line_length must be equal to or less than both length of image_shape,
            but line_length vs image_shape = {line_length} vs {image_shape}"""
            raise ValueError(msg)
        # Store the arguments into member variables.
        self.image_shape = image_shape
        self.line_length = line_length
        self.line_pixel_value = line_pixel_value
        self.min_noise_value = min_noise_value
        self.max_noise_value = max_noise_value

    def __get_all_horizontal_patterns(self) -> np.ndarray:
        """Get all horizontal patterns of the given image shape and length of the line as a flattened array.

        :return np.ndarray: all horizontal patterns as flattened
        """
        # Make the trivial pattern, which the line is set from the head.
        trivial_pattern = np.zeros(self.image_shape[1])
        trivial_pattern[: self.line_length] = self.line_pixel_value
        # Make the patterns for one line.
        num_patterns_for_one_line = self.image_shape[1] - (self.line_length - 1)
        patterns_for_one_line = np.zeros(
            (num_patterns_for_one_line, self.image_shape[1])
        )
        for index in range(num_patterns_for_one_line):
            patterns_for_one_line[index, :] = np.roll(trivial_pattern, index)

        # Put the patterns for one line to every line.
        num_patterns = num_patterns_for_one_line * self.image_shape[0]
        image_length = self.image_shape[0] * self.image_shape[1]
        patterns = np.zeros((num_patterns, image_length))
        for index in range(self.image_shape[0]):
            start_row_index = index * num_patterns_for_one_line
            end_row_index = start_row_index + num_patterns_for_one_line
            start_column_index = index * self.image_shape[1]
            end_column_index = start_column_index + self.image_shape[1]
            patterns[
                start_row_index:end_row_index, start_column_index:end_column_index
            ] = patterns_for_one_line

        return patterns

    def __get_all_vertical_patterns(self) -> np.ndarray:
        """Get all vertical patterns of the given image shape and length of the line as a flattened array by using get_all_horizontal_patterns.

        :return np.ndarray: all vertical patterns as flattened
        """
        # Get all horizontal patterns of the transposed image shape.
        new_image_shape = (self.image_shape[1], self.image_shape[0])
        transposed_patterns = self.__get_all_horizontal_patterns()
        # Transpose each horizontal pattern so that it is the original vertical pattern.
        patterns = np.zeros(transposed_patterns.shape)
        for index, transposed_pattern in enumerate(transposed_patterns):
            reshaped_transposed_pattern = transposed_pattern.reshape(new_image_shape)
            reshaped_pattern = reshaped_transposed_pattern.T
            patterns[index, :] = reshaped_pattern.flatten()

        return patterns

    def generate(self, num_images: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Generate the line dataset.
        The label of horizontal line is -1, otherwise +1.

        :param int num_images: number of images
        :return tuple[list[np.ndarray], list[np.ndarray]]: images and their labels
        """
        # Get all horizontal patterns.
        hor_array = self.__get_all_horizontal_patterns()

        # Create all vertical line patterns.
        ver_array = self.__get_all_vertical_patterns()

        # Generate random images.
        images = []
        labels = []
        for _ in range(num_images):
            rng = qiskit_algorithms.utils.algorithm_globals.random.integers(0, 2)
            if rng == 0:
                labels.append(-1)
                random_image = (
                    qiskit_algorithms.utils.algorithm_globals.random.integers(
                        0, hor_array.shape[0]
                    )
                )
                images.append(np.array(hor_array[random_image]))
            elif rng == 1:
                labels.append(1)
                random_image = (
                    qiskit_algorithms.utils.algorithm_globals.random.integers(
                        0, ver_array.shape[0]
                    )
                )
                images.append(np.array(ver_array[random_image]))

            # Create noise.
            image_length = self.image_shape[0] * self.image_shape[1]
            for i in range(image_length):
                if images[-1][i] == 0:
                    images[-1][i] = (
                        qiskit_algorithms.utils.algorithm_globals.random.uniform(
                            self.min_noise_value, self.max_noise_value
                        )
                    )

        return images, labels
