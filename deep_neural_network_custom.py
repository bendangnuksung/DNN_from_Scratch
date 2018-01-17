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


    def softmax(self,values):
        '''
        Exp(value) / ∑ Exp(value)
        :param value:
        :return:
        '''
        expo = []
        final_value = []
        for val in values:
            expo.append(math.exp(val))
        sum_exp = sum(expo)
        for val in expo:
            final_value.append(val/sum_exp)
        return final_value


    def relu(self,value):
        if (value > 0):
            return value
        else:
            return 0


    def logits_calculation(self, input_values, weight_layer, activation):
        final_logits = []
        for weights in weight_layer:
            logit_sum = 0
            # ∑ input * weights
            for each_input, single_weight in zip(input_values, weights):
                logit_sum += each_input * single_weight
            # activation relu
            if activation == 'relu':
                logit_sum = self.relu(logit_sum)
            final_logits.append(logit_sum)

        # activation softmax for output
        if activation == 'softmax':
            final_logits = self.softmax(final_logits)

        return final_logits



    def feed_forward(self, input_value):
        h_layer_1_value = self.logits_calculation(input_value, self.weight_layer_1, activation='relu')
        h_layer_2_value = self.logits_calculation(h_layer_1_value,self.weight_layer_2, activation='relu')
        output_layer_value = self.logits_calculation(h_layer_2_value, self.weight_output_layer, activation='softmax')


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

