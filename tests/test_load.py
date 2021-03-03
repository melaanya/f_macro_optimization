import unittest

from src.load import load


class EspGameLoad(unittest.TestCase):
    """Tests for ESP Game dataset loading."""

    @classmethod
    def setUpClass(cls):  # noqa
        cls.data = load("./data/ESPGameNN.txt")
        cls.num_classes = 268
        cls.num_docs = 2080

    def test_num_categs(self):  # noqa
        self.assertEqual(len(self.data), self.num_classes)

    def test_num_docs(self):  # noqa
        self.assertEqual(len(self.data[0].pred), self.num_docs)
