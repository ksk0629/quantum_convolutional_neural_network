import numpy as np
import pytest

from src.line_dataset_generator import LineDatasetGenerator


class TestLineDatasetGenerator:

    @classmethod
    def setup_class(cls):
        cls.image_shape = (2, 4)
        cls.line_length = 2
        cls.line_pixel_value = np.pi / 2
        cls.min_noise_value = 0
        cls.max_noise_value = np.pi / 4

        cls.line_dataset = LineDatasetGenerator(
            image_shape=cls.image_shape,
            line_length=cls.line_length,
            line_pixel_value=cls.line_pixel_value,
            min_noise_value=cls.min_noise_value,
            max_noise_value=cls.max_noise_value,
        )

    def test_setup_line_dataset(self):
        """Normal test;

        Check if the instance created in the setup_class function has the same variables as self ones.
        """
        assert self.line_dataset.image_shape == self.image_shape
        assert self.line_dataset.line_length == self.line_length
        assert self.line_dataset.line_pixel_value == self.line_pixel_value
        assert self.line_dataset.min_noise_value == self.min_noise_value
        assert self.line_dataset.max_noise_value == self.max_noise_value

    @pytest.mark.parametrize("num_images", [1, 10, 50, 100])
    def test_generate_with_nomral_args(self, num_images):
        """Normal test;
        Run generate function with the given num_images.

        Check if
        - the length of the return images equals num_images.
        - the length of the return labels equals num_images.
        - the length of each image is the same as one calculated from image_shape.
        - each pixel of all return images is equal to or greater than min_noise_value.
        - each pixel of all return images is equal to or less than max_noise_value.
        - each label is either -1 or +1.
        """
        (images, labels) = self.line_dataset.generate(num_images=num_images)

        # Check if the lengths of images and labels are the same as num_images.
        assert len(images) == num_images == len(labels)
        # Check if the shape of each image is the same as image_shape.
        for image in images:
            assert len(image) == self.image_shape[0] * self.image_shape[1]
        # Check if each pixel is equal to or greater than min_noise_value or equal to line_pixel_value.
        min_noise_flags = np.where(
            np.asarray(images).flatten() >= self.min_noise_value, 1, 0
        )
        line_flags = np.where(
            np.asarray(images).flatten() == self.line_pixel_value,
            1,
            0,
        )
        assert np.all(min_noise_flags + line_flags)
        # Check if each pixel is equal to or less than max_noise_value or equal to line_pixel_value.
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
            LineDatasetGenerator(
                image_shape=image_shape,
                line_length=line_length,
                line_pixel_value=self.line_pixel_value,
                min_noise_value=self.min_noise_value,
                max_noise_value=self.max_noise_value,
            )
