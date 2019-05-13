"""
PLEASE DOCUMENT HERE

Usage: python3 project3.py DATASET.csv
"""

import csv, sys, random, math

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here

class Link: 
    def __init__(self, weight, i, j):
        self.weight = weight
        self.parent = i
        self.child = j

class Node:
    """ class for each node in neural network """
    def __init__(self):
        #each link node
        self.links = []
        #activation value for self

    def __str__(self):
        """ string representation of node """
        pass


class InputNode(Node):
    def __init__(self, value):
        super().__init__()
            self._input = value

    def __str__(self):
        """ string representation of node """
        pass

class OutputNode(Node):
    def __init__(self, value):
        super().__init__()
        self.activation_value = value

    def __str__(self):
        """ string representation of node """
        pass

    def activation_function(self, index):
    """ activation function to get the weighted value of node """
    if len(self.parents) > 0:
        total = 0
        for parent in self.parents:
            total += logistic(parent.)
        return total * self.activation_value
    return logistic(self.activation_value)


class NeuralNetwork:
    """ class for neural network """
    def __init__(self, nodes):
        """ sets up orientation of nodes in network """
        self.network = []
        for i in range(len(nodes)):
            #setting up network size
            self.network.append([])
            for j in range(nodes[i]):
                #adding nodes
                if i == 0:
                    newNode = InputNode(0)
                elif i == len(nodes)-1:
                    newNode = OutputNode(0)
                else:
                    newNode = Node()
                if i < len(nodes)-1:
                    #giving dummy weights to nodes
                    newNode.weights = [1]*nodes[i+1]
                self.network[i].append(newNode)
            if i > 0:
                for node in self.network[i-1]:
                    #setting up links
                    node.links = self.network[i]
                    for link in self.network[i]:
                        #set up reference to parents
                        link.parents.append(node)


    def predict_class(self):
        pass

    def get_outputs(self):
        """ OPTIONAL """
        pass

    def back_propagation_leaning(self, training):
        """ back propagation """
        #assign random weights to nodes
        for layer in self.network:
            for node in layer:
                for i in range(len(node.weights)):
                    node.weights[i] = random.random()
        #propagate forward through network
        self.forward_propagate()
        #errors in outputs
        outputDiffs = []
        for node in self.network[len(self.network)-1]:
            outputDiffs.append()

        for i in range(len(self.network)-2,0,-1):
            pass


    def forward_propagate(self, input):
        """ forward propagation """
        if len(input) != len(self.network[0]):
            raise SyntaxError("input size does not match network size")
        # set input values
        for i in range(len(input)):
            self.network[i].activation_value = input[i]
        # cycle through layers
        for i in range(1, len(self.network)):
            for node in network[i]:
                node.activation_value = activation_function(i)





def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)
    print("pairs:")
    print(pairs)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]

    # Check out the data:
    print("training:")
    for example in training:
        print(example)

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    nn = NeuralNetwork([3, 6, 3])
    # nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()
