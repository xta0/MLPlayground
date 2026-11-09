import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import END_IDX, PAD_IDX, START_IDX, Flickr


class FlickrCaptionLengthTests(unittest.TestCase):
    def test_caption_lengths_and_batching(self):
        vocab = {"<pad>": PAD_IDX, "<start>": START_IDX, "<end>": END_IDX, "<unk>": 3, "word": 4}
        lengths = [0, 1, 49, 50, 51]
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (4, 4)).save(image_path)
            dataset = Flickr([str(image_path)], [[["word"] * length for length in lengths]], vocab)

            for index, length in enumerate(lengths):
                with self.subTest(length=length):
                    image, inputs, targets, mask = dataset[index]
                    sample_length = min(length, 49) + 1
                    self.assertEqual(tuple(image.shape), (3, 128, 128))
                    self.assertEqual(tuple(inputs.shape), (50,))
                    self.assertEqual(tuple(targets.shape), (50,))
                    self.assertEqual(tuple(mask.shape), (50,))
                    self.assertEqual(inputs[0].item(), START_IDX)
                    self.assertEqual(targets[sample_length - 1].item(), END_IDX)
                    self.assertEqual(mask.sum().item(), sample_length)
                    self.assertTrue((inputs[sample_length:] == PAD_IDX).all().item())
                    self.assertTrue((targets[sample_length:] == PAD_IDX).all().item())

            images, inputs, targets, masks = next(iter(DataLoader(dataset, batch_size=len(lengths))))
            self.assertEqual(tuple(images.shape), (5, 3, 128, 128))
            self.assertEqual(tuple(inputs.shape), (5, 50))
            self.assertEqual(tuple(targets.shape), (5, 50))
            self.assertEqual(tuple(masks.shape), (5, 50))


if __name__ == "__main__":
    unittest.main()
