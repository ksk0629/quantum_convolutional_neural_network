import pytest

from src.quant_conv_layer import QuantConvLayer


class TestQuantConvLayer:
    def test_init(self):
        quant_conv_layer = QuantConvLayer()
