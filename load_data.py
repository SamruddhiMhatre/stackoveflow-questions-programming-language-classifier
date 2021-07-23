import tensorflow as tf
import os
import pathlib


class DataLoader:
  def __init__(self, path, filename = None, cachedir = None):
    self.path = path
    self.filename = filename
    self.cachedir = cachedir


  def load_dataset(self):
    dataset = tf.keras.utils.get_file(self.filename, 
                                      self.path, 
                                      untar=True, 
                                      cache_dir= self.cachedir,
                                      cache_subdir = '')

    dataset_dir = pathlib.Path(dataset).parent

    return dataset_dir


  def train_dir(self, dataset_dir, train_path):
    self.dataset_dir = dataset_dir
    self.train_path = train_path
    train_dir = os.path.join(dataset_dir, self.train_path)
    print('Train directory: ', os.listdir(train_dir))

    return train_dir

  def test_dir(self, dataset_dir, test_path):
    self.dataset_dir = dataset_dir
    self.test_path = test_path
    test_dir = os.path.join(dataset_dir, self.test_path)
    print('Test directory: ', os.listdir(test_dir))

    return test_dir
  
  def read_conent(self, directory, content_path):
    '''Returns output of the text files (only)
       in the train/test directory
        
       directory: train/test directory or directory which contents the data(text files)
       content_path: path to the content you want to display'''

    self.directory = directory
    self.content_path = content_path
    sample_file = os.path.join(self.directory, self.content_path)
    with open(sample_file) as f:
      print(f.read())
        
