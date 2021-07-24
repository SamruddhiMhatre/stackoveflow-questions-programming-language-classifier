def compile_model(model, train_data, validation_data, epochs, loss, optimizer, metrics):
  model.compile(loss=loss,
                optimizer=optimizer,
                metrics=metrics)
  
  model_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs)
  
  return model_history





  
