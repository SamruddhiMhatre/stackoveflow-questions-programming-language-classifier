class DataPreprocess:
  def __init__(self):
    pass

  def data_segregation(self, raw_data):
    """
    Storing data separately in the form of text and labels

    Input: Raw Data

    Output: List of Text Batches, List of Label Batches
    """
    
    self.raw_data = raw_data

    texts = []
    labels = []

    for text_batch, label_batch in raw_data:
      texts.append(text_batch)
      labels.append(label_batch)

    return texts, labels
