import random
import math
import csv


train_path = "train_test_data/"
test_path  = "train_test_data/"


def get_train_set():
    """
    training data set should be in CSV ext, last value as output
    :return:
    """
    return


def get_test_set():

    return




class My_DNN():

    epoch = 100

    input_numbers = 5
    h_layer_1_nodes = 5
    h_layer_2_nodes = 3
    output_layer_nodes = 3


    def __init__(self):
        self.weight_layer_1 = self._get_random_weights(self.input_numbers, self.h_layer_1_nodes)
        self.weight_layer_2 = self._get_random_weights(self.h_layer_1_nodes, self.h_layer_2_nodes)
        self.weight_output_layer = self._get_random_weights(self.h_layer_2_nodes, self.output_layer_nodes)



    def _get_random_weights(self, prev_node_count, current_node_count):
        random.seed(10)
        new_weight =[]
        for i in range(current_node_count):
            temp_weight = []
            for j in range(prev_node_count):
                temp_weight.append(random.uniform(-0.3,1))
            new_weight.append(temp_weight)
        return new_weight


    def softmax(self,value):


    def relu(self,value):
        if (value > 0):
            return value
        else:
            return 0


    def call_activation(self,value,activation):
        if activation =='softmax':
            self.softmax(value)

        elif activation =='relu':
            return self.relu(value)

        else:
            return value



    def logits_calculation(self, input_values, weight_layer, activation):
        final_logits = []
        for weights in weight_layer:
            logit_sum = 0
            for each_input, single_weight in zip(input_values, weights):
                logit_sum += each_input * single_weight
            logit_sum = self.call_activation(logit_sum, activation)
            final_logits.append(logit_sum)
        return final_logits



    def feed_forward(self, input_value):
        h_layer_1_value = self.logits_calculation(input_value, self.weight_layer_1, activation='relu')
        h_layer_2_value = self.logits_calculation(h_layer_1_value,self.weight_layer_2, activation='softmax')


    def train(self, train_row):
        input_value = train_row[:-1]
        output_value = train_row[-1]
        output = self.feed_forward(input_value)






if __name__ == '__main__':
    obj = My_DNN()
    train_set = get_train_set()
    test_set = get_test_set()

    for row in train_set:
        obj.train(row)

