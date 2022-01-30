"""picsniff_dataset dataset."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# TODO(picsniff_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(picsniff_dataset): BibTeX citation
_CITATION = """
"""


class PicsniffDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for picsniff_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _decodeLebel(self,labelImageIn,x,y):
    labeMat = labelImageIn[(x*3):(x*3+3),(y*3):(y*3+3),0]
    LabeledScale = (labeMat > 128).astype(int)
    sum =0
    for k in range(3):
        for l in range(3):
            sum += LabeledScale[k,l]
    return sum

  def _cropImage(self,sample_imgIn,x,y):
      return sample_imgIn[(x*48):(x*48+48),(y*48):(y*48+48),0]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(picsniff_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(48, 48, 3)),
            'label': tfds.features.ClassLabel(names=['1','2','3','4','5','7','8']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(picsniff_dataset): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    path = '/home/kisna/Documents/picSniff/dataset_tfds'
    # TODO(picsniff_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path +'/'+ 'train'),
        'val': self._generate_examples(path +'/'+ 'val'),
    }

  def _generate_examples(self, path):
    """Yields examples."""

    # TODO(picsniff_dataset): Yields (key, example) tuples from the dataset
    classNames = os.listdir(path)

    for classNo in classNames:
          imgFolder = path + '/' + classNo + '/'
          arr = os.listdir(imgFolder) 
          for f in arr:
            # print(f)
            yield f, {
                'image': imgFolder + f,
                'label': classNo,
            }


    # # TODO(picsniff_dataset): Yields (key, example) tuples from the dataset
    # for f in path.glob('*.jpeg'):
    #   yield 'key', {
    #       'image': f,
    #       'label': 'yes',
    #   }
