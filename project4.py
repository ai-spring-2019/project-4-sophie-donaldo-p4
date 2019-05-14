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
    def __init__(self, weight, parent, child):
        self.weight = weight
        self.parent = parent
        self.child = child
        self.value = 0

    def activate(self, outgoing):
        return logistic(self.weight * outgoing)

    def __str__(self):
        string = "Link between " + str(self.parent.get_num()) + " and " + str(self.child.get_num()) + " with weight of: " + str(self.weight)
        return string


class Node:
    """ class for each node in neural network """
    def __init__(self, num):
        #each link node
        self.links = []
        #activation value for self
        self.value = 0
        self.num = num

    def __str__(self):
        """ string representation of node """
        pass

    def get_value(self):
        return self.value

    def increase_value(self, new_val):
        self.value += new_val

    def activation_function(self):
        """ activation function to get the weighted value of node """
        for link in links:
            link.parent.increase_value(link.activate(self.value))

    def add_link(self, next_node, weight):
        new_link = Link(weight, self, next_node)
        print(new_link)
        self.links.append(new_link)

    def get_num(self):
        return self.num


class InputNode(Node):
    def __init__(self, num):
        super().__init__(num)
        self.incoming_links = None

    def __str__(self):
        """ string representation of node """
        pass

    def get_input(self, input_val):
        self.value = input_val


class NeuralNetwork:
    """ class for neural network """
    def __init__(self, nodes):
        """ sets up orientation of nodes in network """
        self.network = []
        count = 0
        for i in range(len(nodes)):
            #setting up network size
            self.network.append([])
            for j in range(nodes[i]):
                #adding nodes
                if i == 0:
                    newNode = InputNode(count)
                    count += 1
                else:
                    newNode = Node(count)
                    count += 1
                #if i < len(nodes)-1:
                    # #giving dummy weights to nodes
                    # newNode.weights = [1]*nodes[i+1]
                self.network[i].append(newNode)

        # set up links - hard-coded for one hidden layer
        for input_node in self.network[0]:
            for hidden_node in self.network[1]:
                #initilized with dummy weight of 1
                input_node.add_link(hidden_node, 1)

        for hidden_node in self.network[1]:
            for output_node in self.network[2]:
                #initilized with dummy weight of 1
                hidden_node.add_link(output_node, 1)

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
                for i in range(len(node.links)):
                    node.links[i].weight = random.random()
        #propagate forward through network
        for example in training:
            result = self.forward_propagate(example)
            #errors in outputs
            outputErrors = [[],[]]
            for i in range(len(self.network[len(self.network)-1])):
                error = (example[1][i]-result[i])
                outputErrors[1].append(error)

            for i in range(len(self.network)-2,0,-1):
                for node in self.network[i]:
                    pass
                    #outputErrors[0].append(None)

            #update every weight in network using deltas
            for layer in self.network:
                for node in layer:
                    for link in node.links:
                        link.weight = link.weight #+...


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
