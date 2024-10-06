import numpy as np
import pytest
import src.utils as utils


class TestUtils:

    @classmethod
    def setup_class(cls):
        cls.image_shape = (2, 4)
        cls.line_length = 2
        cls.line_pixel_value = np.pi / 2

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
            image_shape=cls.image_shape,
            line_length=cls.line_length,
            line_pixel_value=cls.line_pixel_value,
        )
        assert np.allclose(correct_example_horizontal_patterns, horizontal_patterns)

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
        - the return array has the given line_pixel_value or zeros.
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
