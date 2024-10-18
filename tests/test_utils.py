import random

import numpy as np
import qiskit_algorithms
import pytest
import torch

import src.utils as utils


class TestUtils:

    @classmethod
    def setup_class(cls):
        cls.image_shape = (2, 4)
        cls.line_length = 2
        cls.line_pixel_value = np.pi / 2
        cls.num_images = 50
        cls.min_noise_value = 0
        cls.max_noise_value = np.pi / 4

        cls.seed = 91

    def test_fix_seed_with_self_args(self):
        """Normal test;
        Run fix_seed and generate random integers through each module and do the same thing.

        Check if the generated integers are the same.
        """
        low = 0
        high = 100000

        utils.fix_seed(self.seed)
        x_random = random.randint(low, high)
        x_qiskit = qiskit_algorithms.utils.algorithm_globals.random.integers(low, high)
        x_np = np.random.randint(low, high)
        x_torch = torch.randint(low=low, high=high, size=(1,))

        utils.fix_seed(self.seed)
        assert x_random == random.randint(low, high)
        assert x_qiskit == qiskit_algorithms.utils.algorithm_globals.random.integers(
            low, high
        )
        assert x_np == np.random.randint(low, high)
        assert x_torch == torch.randint(low=low, high=high, size=(1,))

    def test_get_all_horizontal_patterns_with_self_args(self):
        """Normal test;
        Run get_all_horizontal_patterns with the example arguments,
        which has already set in setup_class.

        Check if the return value is the same as expected.
        """
        # Define the correct horizontal patterns.
        correct_example_horizontal_patterns = np.asarray(
            [
                [np.pi / 2, np.pi / 2, 0, 0, 0, 0, 0, 0],
                [0, np.pi / 2, np.pi / 2, 0, 0, 0, 0, 0],
                [0, 0, np.pi / 2, np.pi / 2, 0, 0, 0, 0],
                [0, 0, 0, 0, np.pi / 2, np.pi / 2, 0, 0],
                [0, 0, 0, 0, 0, np.pi / 2, np.pi / 2, 0],
                [0, 0, 0, 0, 0, 0, np.pi / 2, np.pi / 2],
            ]
        )

        horizontal_patterns = utils.get_all_horizontal_patterns(
            image_shape=self.image_shape,
            line_length=self.line_length,
            line_pixel_value=self.line_pixel_value,
        )
        assert np.allclose(correct_example_horizontal_patterns, horizontal_patterns)

    def test_get_all_vertical_patterns_with_self_args(self):
        """Normal test;
        Run get_all_vertical_patterns with the example arguments,
        which has already set in setup_class.

        Check if the return value is the same as expected.
        """
        # Define the correct vertical patterns.
        correct_example_vertical_patterns = np.asarray(
            [
                [np.pi / 2, 0, 0, 0, np.pi / 2, 0, 0, 0],
                [0, np.pi / 2, 0, 0, 0, np.pi / 2, 0, 0],
                [0, 0, np.pi / 2, 0, 0, 0, np.pi / 2, 0],
                [0, 0, 0, np.pi / 2, 0, 0, 0, np.pi / 2],
            ]
        )
        vertical_patterns = utils.get_all_vertical_patterns(
            image_shape=self.image_shape,
            line_length=self.line_length,
            line_pixel_value=self.line_pixel_value,
        )
        assert np.allclose(correct_example_vertical_patterns, vertical_patterns)

    def test_generate_line_dataset_with_self_args(self):
        """Normal test;
        Run generate_line_dataset with example arguments.

        Check if
        - the length of the return images equals self.num_images.
        - the length of the return labels equals self.num_images.
        - the length of each image is the same as one calculated from self.image_shape.
        - each pixel of all return images is equal to or greater than self.min_noise_value
            or equal to self.line_pixel_value.
        - each pixel of all return images is equal to or less than self.max_noise_value
            or equal to self.line_pixel_value.
        - each label is either -1 or +1.
        """
        (images, labels) = utils.generate_line_dataset(
            num_images=self.num_images,
            image_shape=self.image_shape,
            line_length=self.line_length,
            line_pixel_value=self.line_pixel_value,
            min_noise_value=self.min_noise_value,
            max_noise_value=self.max_noise_value,
        )

        # Check if the lengths of images and labels are the same as self.num_images.
        assert len(images) == self.num_images == len(labels)
        # Check if the shape of each image is the same as self.image_shape.
        for image in images:
            assert len(image) == self.image_shape[0] * self.image_shape[1]
        # Check if each pixel is equal to or greater than self.min_noise_value or equal to self.line_pixel_value.
        min_noise_flags = np.where(
            np.asarray(images).flatten() >= self.min_noise_value, 1, 0
        )
        line_flags = np.where(
            np.asarray(images).flatten() == self.line_pixel_value,
            1,
            0,
        )
        assert np.all(min_noise_flags + line_flags)
        # Check if each pixel is equal to or less than self.max_noise_value or equal to self.line_pixel_value.
        max_noise_flags = np.where(
            np.asarray(images).flatten() <= self.max_noise_value,
            1,
            0,
        )
        assert np.all(max_noise_flags + line_flags)
        # Check if each label is either -1 or +1.
        minus_zero_flags = np.where(np.asarray(labels) == -1, 1, 0)
        plus_zero_flags = np.where(np.asarray(labels) == +1, 1, 0)
        assert np.all(minus_zero_flags + plus_zero_flags)

    @pytest.mark.parametrize(
        "image_shape_and_line_length",
        [((2, 4), 5), ((2, 5), 6), ((3, 4), 5), ((10, 5), 6)],
    )
    def test_get_all_horizontal_patterns_with_too_long_line_length(
        self, image_shape_and_line_length
    ):
        """Abnormal test;
        Run get_all_horizontal_patterns with a certain image_shape and line_length,
        which is longer than image_shape[1] (= columns).

        Check if ValueError happens.
        """
        (image_shape, line_length) = image_shape_and_line_length
        with pytest.raises(ValueError):
            utils.get_all_horizontal_patterns(
                image_shape=image_shape,
                line_length=line_length,
                line_pixel_value=self.line_pixel_value,
            )

    @pytest.mark.parametrize("line_pixel_value", [1, -1, 1.1, np.pi])
    @pytest.mark.parametrize(
        "image_shape_and_line_length",
        [((1, 6), 3), ((2, 2), 2), ((3, 9), 2), ((4, 2), 1)],
    )
    def test_get_all_horizontal_patterns_with_non_default_args(
        self, line_pixel_value, image_shape_and_line_length
    ):
        """Normal test;
        Run get_all_horizontal_patterns with non default arguments

        Check if
        - each pattern in the returned array has the same number of elements
            as the one calculated from the given image_shape.
        each element of the return array is either the given line_pixel_value or zeros.
        - each pattern can be reshaped to the given image_shape.
        - the summation of each row of reshaped pattern is either zero
            or line_pixel_value * line_length.
        - each row of reshaped pattern has only one row
            whose summation is line_pixel_value * line_length and others are zero.
        """
        (image_shape, line_length) = image_shape_and_line_length
        patterns = utils.get_all_horizontal_patterns(
            image_shape=image_shape,
            line_length=line_length,
            line_pixel_value=line_pixel_value,
        )
        for pattern in patterns:
            # Check if the number of elements are the same as one calculated from image_shape.
            assert len(pattern) == image_shape[0] * image_shape[1]

            # Check if each element is either zero or line_pixel_value.
            for pixel in pattern:
                assert pixel == 0 or pixel == line_pixel_value

            # Get the reshaped pattern.
            reshaped_pattern = pattern.reshape(image_shape)

            non_zero_line_count = 0
            for reshaped_line in reshaped_pattern:
                # Check if the summation of each row of the reshaped pattern is either
                # zero or line_pixel_value * line_length.
                assert (
                    np.sum(reshaped_line) == line_pixel_value * line_length
                    or np.sum(reshaped_line) == 0
                )
                if np.sum(reshaped_line) == line_pixel_value * line_length:
                    non_zero_line_count += 1
            # Check if there is only one row whose the summation is line_pixel_value * line_length.
            assert non_zero_line_count == 1

    @pytest.mark.parametrize(
        "image_shape_and_line_length",
        [((2, 4), 3), ((2, 5), 4), ((3, 4), 4), ((10, 5), 11)],
    )
    def test_get_all_vertical_patterns_with_too_long_line_length(
        self, image_shape_and_line_length
    ):
        """Abnormal test;
        Run get_all_vertical_patterns with a certain image_shape and line_length,
        which is longer than image_shape[0] (= rows).

        Check if ValueError happens.
        """
        (image_shape, line_length) = image_shape_and_line_length
        with pytest.raises(ValueError):
            utils.get_all_vertical_patterns(
                image_shape=image_shape,
                line_length=line_length,
                line_pixel_value=self.line_pixel_value,
            )

    @pytest.mark.parametrize("line_pixel_value", [1, -1, 1.1, np.pi])
    @pytest.mark.parametrize(
        "image_shape_and_line_length",
        [((2, 1), 2), ((2, 2), 2), ((3, 9), 2), ((4, 2), 3)],
    )
    def test_get_all_vertical_patterns_with_non_default_args(
        self, line_pixel_value, image_shape_and_line_length
    ):
        """Normal test;
        Run get_all_vertical_patterns with non default arguments

        Check if
        - each pattern in the returned array has the same number of elements
            as the one calculated from the given image_shape.
        - each element of the return array is either the given line_pixel_value or zeros.
        - each pattern can be reshaped to the given image_shape.
        - the summation of each column of reshaped pattern is either zero
            or line_pixel_value * line_length.
        - each column of reshaped pattern has only one column
            whose summation is line_pixel_value * line_length and others are zero.
        """
        (image_shape, line_length) = image_shape_and_line_length
        patterns = utils.get_all_vertical_patterns(
            image_shape=image_shape,
            line_length=line_length,
            line_pixel_value=line_pixel_value,
        )
        for pattern in patterns:
            # Check if the number of elements are the same as one calculated from image_shape.
            assert len(pattern) == image_shape[0] * image_shape[1]

            # Check if each element is either zero or line_pixel_value.
            for pixel in pattern:
                assert pixel == 0 or pixel == line_pixel_value

            # Get the reshaped pattern.
            reshaped_pattern = pattern.reshape(image_shape)
            # Transpose the pattern to check vertical line.
            transpoed_reshaped_pattern = reshaped_pattern.T

            non_zero_line_count = 0
            for reshaped_line in transpoed_reshaped_pattern:
                # Check if the summation of each row of the reshaped pattern is either
                # zero or line_pixel_value * line_length.
                assert (
                    np.sum(reshaped_line) == line_pixel_value * line_length
                    or np.sum(reshaped_line) == 0
                )
                if np.sum(reshaped_line) == line_pixel_value * line_length:
                    non_zero_line_count += 1
            # Check if there is only one row whose the summation is line_pixel_value * line_length.
            assert non_zero_line_count == 1

    @pytest.mark.parametrize(
        "image_shape_and_line_length",
        [((2, 4), 3), ((3, 3), 4), ((4, 2), 3), ((10, 12), 11), ((12, 10), 11)],
    )
    def test_generate_line_dataset_with_too_long_line_length(
        self, image_shape_and_line_length
    ):
        """Abnormal test;
        Run generate_line_dataset with line_length that is greater than
        either image_shape[0] or image_shape[1].

        Check if ValueError happens.
        """
        (image_shape, line_length) = image_shape_and_line_length
        with pytest.raises(ValueError):
            utils.generate_line_dataset(
                num_images=self.num_images,
                image_shape=image_shape,
                line_length=line_length,
                line_pixel_value=self.line_pixel_value,
                min_noise_value=self.min_noise_value,
                max_noise_value=self.max_noise_value,
            )

    @pytest.mark.parametrize("num_images", [1, 100, 200])
    @pytest.mark.parametrize(
        "image_shape_and_line_length",
        [((2, 2), 2), ((3, 9), 2), ((4, 2), 2)],
    )
    @pytest.mark.parametrize("line_pixel_value", [1, -1, 1.1, np.pi])
    @pytest.mark.parametrize(
        "min_and_max_noise_values", [(-100, 100), (0, 0), (-10, 0), (0, 10)]
    )
    def test_generate_line_dataset_with_non_default_args(
        self,
        num_images,
        image_shape_and_line_length,
        line_pixel_value,
        min_and_max_noise_values,
    ):
        """Normal test;
        Run generate_line_dataset with non default arguments.

        Check if
        - the length of the return images equals num_images.
        - the length of the return labels equals num_images.
        - the length of each image is the same as one calculated from image_shape.
        - each pixel of all return images is equal to or greater than min_noise_value.
        - each pixel of all return images is equal to or less than max_noise_value.
        - each label is either -1 or +1.
        """
        (image_shape, line_length) = image_shape_and_line_length
        (min_noise_value, max_noise_value) = min_and_max_noise_values
        (images, labels) = utils.generate_line_dataset(
            num_images=num_images,
            image_shape=image_shape,
            line_length=line_length,
            line_pixel_value=line_pixel_value,
            min_noise_value=min_noise_value,
            max_noise_value=max_noise_value,
        )

        # Check if the lengths of images and labels are the same as num_images.
        assert len(images) == num_images == len(labels)
        # Check if the shape of each image is the same as image_shape.
        for image in images:
            assert len(image) == image_shape[0] * image_shape[1]
        # Check if each pixel is equal to or greater than min_noise_value or equal to line_pixel_value.
        min_noise_flags = np.where(
            np.asarray(images).flatten() >= min_noise_value, 1, 0
        )
        line_flags = np.where(
            np.asarray(images).flatten() == line_pixel_value,
            1,
            0,
        )
        assert np.all(min_noise_flags + line_flags)
        # Check if each pixel is equal to or less than max_noise_value or equal to line_pixel_value.
        max_noise_flags = np.where(
            np.asarray(images).flatten() <= max_noise_value,
            1,
            0,
        )
        assert np.all(max_noise_flags + line_flags)
        # Check if each label is either -1 or +1.
        minus_zero_flags = np.where(np.asarray(labels) == -1, 1, 0)
        plus_zero_flags = np.where(np.asarray(labels) == +1, 1, 0)
        assert np.all(minus_zero_flags + plus_zero_flags)

    @pytest.mark.parametrize(
        "image_shape_and_line_length",
        [((2, 2), 2), ((3, 9), 2), ((4, 2), 2)],
    )
    def test_generate_line_dataset_with_fixed_noise_0(
        self,
        image_shape_and_line_length,
    ):
        """Normal test;
        Run generate_line_dataset with fixed noise value 0,
        which allows us to check the line is valid.

        Check if
        - the length of the return images equals num_images.
        - the length of the return labels equals num_images.
        - the length of each image is the same as one calculated from image_shape.
        - each pixel of all return images is equal to or greater than min_noise_value.
        - each pixel of all return images is equal to or less than max_noise_value.
        - each image has only one line.
        - each label is either -1 or +1.
        """
        (image_shape, line_length) = image_shape_and_line_length
        min_noise_value = 0
        max_noise_value = 0
        (images, labels) = utils.generate_line_dataset(
            num_images=self.num_images,
            image_shape=image_shape,
            line_length=line_length,
            line_pixel_value=self.line_pixel_value,
            min_noise_value=min_noise_value,
            max_noise_value=max_noise_value,
        )

        # Check if the lengths of images and labels are the same as self.num_images.
        assert len(images) == self.num_images == len(labels)
        # Check if the shape of each image is the same as image_shape.
        for image in images:
            assert len(image) == image_shape[0] * image_shape[1]
        # Check if each pixel is equal to or greater than min_noise_value or equal to self.line_pixel_value.
        min_noise_flags = np.where(
            np.asarray(images).flatten() >= min_noise_value, 1, 0
        )
        line_flags = np.where(
            np.asarray(images).flatten() == self.line_pixel_value,
            1,
            0,
        )
        assert np.all(min_noise_flags + line_flags)
        # Check if each pixel is equal to or less than max_noise_value or equal to self.line_pixel_value.
        max_noise_flags = np.where(
            np.asarray(images).flatten() <= max_noise_value,
            1,
            0,
        )
        assert np.all(max_noise_flags + line_flags)
        # Check if each label is either -1 or +1.
        minus_zero_flags = np.where(np.asarray(labels) == -1, 1, 0)
        plus_zero_flags = np.where(np.asarray(labels) == +1, 1, 0)
        assert np.all(minus_zero_flags + plus_zero_flags)
        # Check if each image has only one line.
        for image in images:
            reshaped_image = image.reshape(image_shape)
            # Apply for-loop to each row to see if there is a horizontal line.
            horizontal_line_count = 0
            for row in reshaped_image:
                if np.sum(row) == line_length * self.line_pixel_value:
                    horizontal_line_count += 1
            # Apply for-loop to each row to see if there is a vertical line.
            vertical_line_count = 0
            for column in reshaped_image.T:
                if np.sum(column) == line_length * self.line_pixel_value:
                    vertical_line_count += 1
            assert horizontal_line_count + vertical_line_count == 1
