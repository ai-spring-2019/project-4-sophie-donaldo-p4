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
class Node:
    """ class for each node in neural network """
    def __init__(self):
        #weights for each link node
        self.weights = []
        #each link node
        self.links = []
        #reference to parent nodes for activation function
        self.parents = []
        #activation value for self
        self.activation_value = 1

    def __str__(self):
        """ string representation of node """
        return "Weigts:\n"+\
        str(self.weights)+\
        "\nLinks:\n"+\
        str(self.links)+\
        "\nParents:\n"+\
        str(self.parents)

    def activation_function(self, index):
        """ activation function to get the weighted value of node """
        if len(self.parents) > 0:
            total = 0
            for parent in self.parents:
                #recursively call parent activation functions
                total += parent.activation_function(index)
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
        pass

    def forward_propagate(self, x):
        pass

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
    nn = NeuralNetwork([6, 6])
    # nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()
