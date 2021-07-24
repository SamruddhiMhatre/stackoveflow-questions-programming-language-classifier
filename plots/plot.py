import matplotlib.pyplot as plt


def plot(history_dict, value):
  acc = history_dict['accuracy']
  val_acc = history_dict['val_accuracy']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']
  epochs = range(1, len(acc) + 1)

  if value == 'loss':
    train_plot = loss
    val_plot = val_loss
  elif value == 'accuracy':
    train_plot = acc
    val_plot = val_acc

  # "bo" is for "blue dot"
  plt.plot(epochs, train_plot, 'bo', label='Training ' + value)
  # b is for "solid blue line"
  plt.plot(epochs, val_plot, 'b', label='Validation ' + value)
  plt.title('Training and validation ' + value)
  plt.xlabel('Epochs')
  plt.ylabel(value)
  plt.legend()

  plt.show()
