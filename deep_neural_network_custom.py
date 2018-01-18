import random
import math
import csv



##############################################
# User define (change accordingly)
train_path = "train_test_data/iris.csv"
test_path  = "train_test_data/iris.csv"
input_layer = 4
output_layer =3

# optional
learning_rate = 0.001
epoch = 100

##############################################





def one_hot_encoding(value):
    one_hot = []
    for i in range(output_layer):
        if (i+1) == value:
            one_hot.append(1)
        else:
            one_hot.append(0)
    return one_hot


def get_train_set(path):
    """
        1) training data set should be in CSV format
        2) last value of the row should be output(class)
        3) output(class) should be labelled numeric eg: 1,2,3
    """
    file = open(path)
    next(file)
    file = csv.reader(file)
    rows = []
    for row in file:
        row = list(map(float,row))
        row[-1] = one_hot_encoding(row[-1])
        rows.append(row)
    return rows



def get_test_set():

    return


def softmax(values):
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


def relu(value):
    if (value > 0):
        return value
    else:
        return 0


class My_DNN():

    h_layer_1_nodes = 5
    h_layer_2_nodes = 3
    output_layer_nodes = 3


    def __init__(self, input_numbers, output_number, learning_rate= 0.001, epoch=200):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.input_numbers = input_numbers
        self.output_layer_nodes = output_number
        self.weight_layers = []
        self.weight_layers.append(self._get_random_weights(self.input_numbers, self.h_layer_1_nodes))
        self.weight_layers.append(self._get_random_weights(self.h_layer_1_nodes, self.h_layer_2_nodes))
        self.weight_layers.append(self._get_random_weights(self.h_layer_2_nodes, self.output_layer_nodes))

    def _get_random_weights(self, prev_node_count, current_node_count):
        random.seed(5)
        new_weight =[]
        for i in range(current_node_count):
            temp_weight = []
            for j in range(prev_node_count):
                temp_weight.append(random.uniform(-0.3,1))
            new_weight.append(temp_weight)
        return new_weight


    def cross_entropy(self, output, expected_output):
        cost = 0
        for i in range(len(output)):
            if expected_output[i] == 1:
                cost -= math.log(output[i])
            else:
                cost -= math.log(1- output[i])
        return cost



    def logits_calculation(self, input_values, weight_layer, activation ='relu'):
        final_logits = []
        for weights in weight_layer:
            logit_sum = 0.0
            # ∑ input * weights
            for each_input, single_weight in zip(input_values, weights):
                logit_sum += each_input * single_weight
            # activation relu
            if activation == 'relu':
                logit_sum = relu(logit_sum)
            final_logits.append(logit_sum)

        return final_logits


    def feed_forward(self, input_value):
        self.h_layer_input_value = []
        self.h_layer_input_value.append(self.logits_calculation(input_value, self.weight_layers[0]))
        self.h_layer_input_value.append(self.logits_calculation(self.h_layer_input_value[0],self.weight_layers[1]))
        self.h_layer_input_value.append(self.logits_calculation(self.h_layer_input_value[1], self.weight_layers[2]))

        return softmax(self.h_layer_input_value[2])


    def back_propogation(self,cost):

        for i,weights in enumerate(self.weight_layers):
            for j,weight in enumerate(weights):
                for k, each_weight in enumerate(weight):
                    original_weight = each_weight
                    self.weight_layers[i][j][k] += learning_rate
                    output = self.feed_forward(self.input_value)
                    new_cost = self.cross_entropy(output,self.expected_value)
                    if new_cost == cost:
                        self.weight_layers[i][j][k] = original_weight
                    elif new_cost > cost:
                        self.weight_layers[i][j][k] = original_weight - learning_rate
                        cost = cost - (new_cost-cost)
                    else:
                        cost = new_cost


    def train(self, train_row):
        self.input_value = train_row[:-1]
        self.expected_value = train_row[-1]
        for i in range(epoch):
            output = self.feed_forward(self.input_value)
            cost = self.cross_entropy(output, self.expected_value)
            self.back_propogation(cost)
            print(cost)
            print (i)


    def test(self, test_row):
        input_value = test_row[:-1]
        expected_value = test_row[-1]
        output = self.feed_forward(input_value)
        print ("expected: ", expected_value)
        print("result :", output)




if __name__ == '__main__':
    obj = My_DNN(input_layer,output_layer)
    train_set = get_train_set(train_path)
    test_set = get_test_set()

    for row in train_set:
        obj.train(row)
        break

    for row in train_set:
        obj.test(row)
        break

