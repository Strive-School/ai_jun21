
import data_handler as dh

x_train, x_test, y_train, y_test = dh.load_data("data/turkish_stocks.csv", 7)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

for epochs in range(2):
    x_train_batch, x_test_batch, y_train_batch, y_test_batch = dh.to_batches(x_train, x_test, y_train, y_test, 7)

    print(x_train_batch.shape, x_test_batch.shape, y_train_batch.shape, y_test_batch.shape)