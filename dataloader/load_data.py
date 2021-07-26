import tensorflow as tf
import os
import pathlib
from tensorflow.keras import preprocessing

class DataLoader:
  """
  Data loading class: From loading data from a given path to 
  preparation of train,test and validation data
  """
  
  def __init__(self, path, filename = None, cachedir = None):
    self.path = path
    self.filename = filename
    self.cachedir = cachedir


  def load_dataset(self):
    """
    Load data from the given path
    """
    
    dataset = tf.keras.utils.get_file(self.filename, 
                                      self.path, 
                                      untar=True, 
                                      cache_dir= self.cachedir,
                                      cache_subdir = '')

    dataset_dir = pathlib.Path(dataset).parent

    return dataset_dir


  def train_dir(self, dataset_dir, train_path):
    """ 
    Load training data's directory
    """
    
    self.dataset_dir = dataset_dir
    self.train_path = train_path
    train_dir = os.path.join(dataset_dir, self.train_path)
    print('Train directory: ', os.listdir(train_dir))

    return train_dir


  def test_dir(self, dataset_dir, test_path):
    """
    Load testing data's directory
    """
    
    self.dataset_dir = dataset_dir
    self.test_path = test_path
    test_dir = os.path.join(dataset_dir, self.test_path)
    print('Test directory: ', os.listdir(test_dir))

    return test_dir

  
  def read_content(self, directory, content_path):
    '''Returns output of the text files (only)
       in the train/test directory
        
       directory: train/test directory or directory which contents the data(text files)
       content_path: path to the content you want to display'''

    self.directory = directory
    self.content_path = content_path
    sample_file = os.path.join(self.directory, self.content_path)
    with open(sample_file) as f:
      print(f.read())


  def load_train_data(self, train_dir, validation_split, batch_size=None, seed=None):
    """
    Prepare train data
    """
    self.train_dir = train_dir
    self.validation_split = validation_split
    self.batch_size = batch_size
    self.seed = seed
    
     
    # raw train data 
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        self.train_dir,
        labels='inferred',
        seed=self.seed,
        batch_size=self.batch_size,
        validation_split=self.validation_split,
        subset = 'training')

    print('Number of batches in training set: ', tf.data.experimental.cardinality(raw_train_ds), '\n')

    return raw_train_ds


  def load_validation_data(self, train_dir, validation_split, batch_size=None, seed=None):
    """
    Prepare validation data
    """
    self.train_dir = train_dir
    self.batch_size = batch_size
    self.seed = seed
    self.validation_split = validation_split

    # raw validation data
    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        self.train_dir,
        labels='inferred',
        seed=self.seed,
        batch_size=self.batch_size,
        validation_split=self.validation_split,
        subset = 'validation')

    print('Number of batches in validation set: ', tf.data.experimental.cardinality(raw_val_ds),'\n')

    return raw_val_ds


  def load_test_data(self, test_dir, batch_size):
    """
    Prepare test data
    """
    self.test_dir = test_dir
    self.batch_size = batch_size

    # raw test data
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    self.test_dir,
    batch_size = self.batch_size)
    
    print('Number of batches in testing set: ', tf.data.experimental.cardinality(raw_test_ds))

    return raw_test_ds

        
