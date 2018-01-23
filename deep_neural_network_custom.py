import random
import math
import csv



##############################################
# User define (change accordingly)
train_path = "train_test_data/iris.csv"
test_path  = "train_test_data/test.csv"
input_layer = 4
output_layer = 3

# optional
learning_rate = 0.001
epoch = 50

##############################################


def one_hot_encoding(value):
    """
    encodes categorical integer. 1:100, 2:010, 3:001
    :param value:
    :return:
    """
    one_hot = []
    for i in range(output_layer):
        if (i+1) == value:
            one_hot.append(1)
        else:
            one_hot.append(0)
    return one_hot


def get_dataset(path, is_type):
    """
        1) training data set should be in CSV format
        2) last value of the row should be output(class)
        3) output(class) should be labelled numeric eg: 1,2,3
    """
    file = open(path)
    if is_type == 'train':
        next(file)
    file = csv.reader(file)
    rows = []
    for row in file:
        row = list(map(float, row))
        row[-1] = one_hot_encoding(row[-1])
        rows.append(row)
    return rows

# Activation and derivative functions:
def softmax(values):
    expo = []
    final_value = []
    for val in values:
        expo.append(math.exp(val))
    sum_exp = sum(expo)
    for val in expo:
        final_value.append(val/sum_exp)
    return final_value

def relu(value):
    return value * (value>0)

def sigmoid(value):
    return 1 / (1 + math.exp(-value))

def sigmoid_derivative(value):
    return value * (1.0 - value)



class My_DNN():

    h_layer_1_nodes = 20
    h_layer_2_nodes = 20

    def __init__(self, input_numbers, output_number, learning_rate= 0.001, epoch=10, layers_no=2):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.input_numbers = input_numbers
        self.output_numbers = output_number
        self.hidden_layers = []
        for i in range(layers_no+1):
            self.hidden_layers.append([])
            if i == 0:
                self.hidden_layers[0] = (self._get_random_weights(self.input_numbers, self.h_layer_1_nodes))
            elif i == layers_no:
                self.hidden_layers[i] = (self._get_random_weights(self.h_layer_1_nodes, self.output_numbers))
            else:
                self.hidden_layers[i] = (self._get_random_weights(self.h_layer_1_nodes, self.h_layer_2_nodes))


    def _get_random_weights(self, prev_node_count, current_node_count):
        new_weight =[]
        for i in range(current_node_count):
            temp_weight = []
            for j in range(prev_node_count):
                temp_weight.append(random.uniform(0, 1))
            new_weight.append({'weights':temp_weight})
        return new_weight


    def cross_entropy(self, output, expected_output):
        cost = 0
        for i in range(len(output)):
            if output[i] == 0:
                continue
            if expected_output[i] == 1:
                cost -= math.log(output[i])
            else:
                cost -= math.log(1- output[i])
        return cost


    def logits_calculation(self, input_values, layer, layer_no, activation ='sigmoid'):
        final_logits = []
        for i,neuron in enumerate(layer):
            logit_sum = 0.0
            # ∑ input * weights
            for each_input, each_weight in zip(input_values, neuron['weights']):
                logit_sum += each_input * each_weight
            if activation == 'sigmoid':
                logit_sum = sigmoid(logit_sum)
                self.hidden_layers[layer_no][i]['output'] = logit_sum
            elif activation == 'relu':
                logit_sum = relu(logit_sum)
                self.hidden_layers[layer_no][i]['output'] = logit_sum
            final_logits.append(logit_sum)
        if activation == 'softmax':
            softmax_result = softmax(final_logits)
            for i in range(len(softmax_result)):
                self.hidden_layers[layer_no][i]['output'] = softmax_result[i]
            return softmax_result
        return final_logits


    def feed_forward(self, input_value):
        output_1 = self.logits_calculation(input_value, self.hidden_layers[0], layer_no=0)
        output_2 = self.logits_calculation(output_1, self.hidden_layers[1], layer_no=1)
        output_3 = self.logits_calculation(output_2, self.hidden_layers[2], layer_no=2, activation='softmax')
        return output_3


    def update_weights(self):
        for i in range(len(self.hidden_layers)):
            inputs = self.input_value
            if i != 0:
                inputs = [neuron['output'] for neuron in self.hidden_layers[i - 1]]
            for j in range(len(self.hidden_layers[i])):
                for k in range(len(inputs)):
                    self.hidden_layers[i][j]['weights'][k] += self.learning_rate * self.hidden_layers[i][j]['delta'] * inputs[k]
                self.hidden_layers[i][j]['weights'][-1] += self.learning_rate * self.hidden_layers[i][j]['delta']


    def back_propogation_1(self):
        for i in reversed(range(len(self.hidden_layers))):
            layer = self.hidden_layers[i]
            errors = []
            if i == len(self.hidden_layers) - 1:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(self.expected_value[j] - neuron['output'])
            else:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.hidden_layers[i+1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])
        self.update_weights()


    def test(self, test_row):
        """
        gets percentage of each input to occur for test purpose
        :param test_row:
        :return:
        """
        input_value = test_row[:-1]
        expected_value = test_row[-1]
        output = self.feed_forward(input_value)
        print("expected: ", expected_value)
        print("result :", output)


    def accuracy(self, test_rows):
        correct = 0
        for row in test_rows:
            input_value = row[:-1]
            expected_value = row[-1]
            output_layer = self.feed_forward(input_value)
            max_prob = max(output_layer)
            for i,j in zip(output_layer,expected_value):
                if i == max_prob and j == 1:
                    correct +=1
        accuracy = correct/len(test_rows)
        return accuracy*100


    def train(self, train_row):
        self.input_value = train_row[:-1]
        self.expected_value = train_row[-1]
        cost_list = []
        for i in range(epoch):
            output = self.feed_forward(self.input_value)
            cost = self.cross_entropy(output, self.expected_value)
            cost_list.append(cost)
            if i%5 == 0:
                print(sum(cost_list)/len(cost_list))
            self.back_propogation_1()
        pass


if __name__ == '__main__':
    obj = My_DNN(input_layer,output_layer, learning_rate= learning_rate)
    train_set = get_dataset(train_path, is_type= 'train')
    test_set = get_dataset(test_path, is_type= 'test')
    for i,row in enumerate(train_set):
        obj.train(row)

    for i,row in enumerate(test_set):
        obj.test(row)

    print ("accuracy: ",obj.accuracy(test_set),"%")