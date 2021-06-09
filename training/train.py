import tensorflow as tf

from training.train_and_test import train_and_test


if __name__ == "__main__": 

  EPOCHS = 5

  for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_and_test.train_loss.reset_states()
    train_and_test.train_accuracy.reset_states()
    train_and_test.test_loss.reset_states()
    train_and_test.test_accuracy.reset_states()

    for images, labels in train_ds:
      train_and_test.train_step(images, labels)


    print(
      f'Epoch {epoch + 1}, '
      f'Loss: {train_and_test.train_loss.result()}, '
      f'Accuracy: {train_and_test.train_accuracy.result() * 100}, '
    )