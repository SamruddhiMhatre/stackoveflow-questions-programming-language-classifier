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

  
  def text_standardization(self, text):
    """
    Input: Batch Text list

    Output: Returns list after removing the html tags from the lower cased text
    """

    self.text = text
    text = tf.strings.lower(text)
    cleantext = tf.strings.regex_replace(text,"<[^>]+>", " ")

    return cleantext


  
  def vectorize_text(self, max_features,text, label):

    self.max_features = max_features
    self.text = text
    self.label = label
  
    vectorize_layer = TextVectorization(   # for batch of strings
    standardize=self.text_standardization,
    max_tokens=max_features,
    output_mode='binary'              #  binary mode to build bag of words model
    )

    vectorize_layer.adapt(self.text)

    text = tf.expand_dims(self.text, -1)
    
    return vectorize_layer(text), self.label
