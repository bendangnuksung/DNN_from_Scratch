import random
import math

train_path = "train_test_data/"
test_path  = "train_test_data/"



def get_train_set():
    pass



def get_test_set():
    pass


class My_DNN():

    epoch = 100

    input_numbers = 5
    layer_1_nodes = 5
    layer_2_nodes = 3
    output_layer_nodes = 3


    def __init__(self):
        self.weight_layer_1 = self._get_random_weights(self.input_numbers, self.layer_1_nodes)
        self.weight_layer_2 = self._get_random_weights(self.layer_1_nodes, self.layer_2_nodes)
        self.weight_output_layer = self._get_random_weights(self.layer_2_nodes, self.output_layer_nodes)



    def _get_random_weights(self, prev_node_count, current_node_count):
        random.seed(10)
        new_weight =[]
        for i in range(current_node_count):
            temp_weight = []
            for j in range(prev_node_count):
                temp_weight.append(random.uniform(-0.3,1))
            new_weight.append(temp_weight)
        return new_weight


    def train(self):





if __name__ == '__main__':
    obj = My_DNN()
    train_set = get_train_set()
    test_set = get_test_set()

