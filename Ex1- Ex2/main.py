
import load_data as ld

from neural_network import Network

training_data, validation_data, test_data = ld.load_data()

# Each training and test example is assigned to one of the following labels:

# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot

network = Network([784, 30, 30, 10])

# training_data, epochs, mini_batch_size, learning_rate, validation_data=None, test_data=None
network.SGD(training_data, 120, 300, 0.4,
            validation_data=validation_data, test_data=test_data)
