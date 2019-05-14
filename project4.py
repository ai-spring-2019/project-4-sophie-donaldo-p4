"""
Project 3
Sophie Menashi + Donald Holley 
Project 4
Usage: python3 project3.py DATASET.csv
"""

import csv, sys, random, math, copy, itertools

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

def logistic_deriv(x):
    """Logistic / sigmoid function"""
    try:
        denom = (math.e ** x + 1) **2
    except OverflowError:
        return 0.0
    return math.e ** -x / denom

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
        nn.forward_propagate(x[1:])
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)
        #print(true_positives)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here

class Link:
    """Class for storing attributes of links"""
    def __init__(self, weight, parent, child):
        self.weight = weight
        self.parent = parent
        self.child = child

    def weight_value(self, value):
        """method for weighting a value travelling through link"""
        return value * self.weight

    def __str__(self):
        string = "Link between " + str(self.parent.get_num()) + " and " + str(self.child.get_num()) + " with weight of: " + str(self.weight)
        return string


class Node:
    """ class for each node in neural network """
    def __init__(self, num):
        self.links = []
        self.value = 0
        self.num = num
        self.error = 0

    def __str__(self):
        """ string representation of node """
        pass

    def get_value(self):
        return self.value

    def increase_value(self, new_val):
        self.value += new_val

    def propogate(self):
        """ propogates value to all child nodes"""
        for link in self.links:
            #if outgoing link
            if link.parent.get_num() == self.num:
                link.child.increase_value(link.weight_value(self.value))

    def activate(self):
        """applies application function to incoming value"""
        self.value = logistic(self.value)

    def sum_outgoing_weights(self):
        """sums all outgoing weights for back_prop"""
        total = 0
        for link in self.links:
            if link.parent.get_num() == self.num:
                total += (link.weight * link.child.error)
        return total

    def add_link(self, next_node, weight):
        """adds outgoing link"""
        new_link = Link(weight, self, next_node)
        #print(new_link)
        self.links.append(new_link)

    def get_num(self):
        """returns num"""
        return self.num


class InputNode(Node):
    """A Node subclass for input nodes"""
    def __init__(self, num):
        super().__init__(num)
        self.incoming_links = None

    def __str__(self):
        """ string representation of node """
        pass

    def set_input(self, input_val):
        """sets value to input value"""
        self.value = input_val


class NeuralNetwork:
    """ class for neural network """
    def __init__(self, nodes):
        """ sets up orientation of nodes in network """
        self.errors = []
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

        # set up links
        for layer in range(1, len(self.network) - 1):
            for node in self.network[layer]:
                for next_node in self.network[layer + 1]:
                    #initilized with dummy weight of 1
                    node.add_link(next_node, 1)

    def predict_class(self):
        """Predicts the class for classifcaiton problems"""
        output_val = self.network[len(self.network) -1][0].get_value
        if output_val == 0:
            return 0.0
        else:
            return 1.0

        # for 3-class dataset 
        # 
    #     if output_val < 0.33:
    #         return 0
    #     elif output_val < 0.66:
    #         return 1
    #     else:
    #         return 2

    def back_propagation_learning(self, training):
        """ back propagation """
        
        #assign random weights to nodes
        for layer in self.network:
            for node in layer:
                for i in range(len(node.links)):
                    node.links[i].weight = random.random()
            #propagate forward through network
        for _ in range(100):
            for example in training:
                #print(example[0][1:])
                result = self.forward_propagate(example[0][1:])
                #errors in outputs
                for i in range(len(self.network[-1])):
                    error = logistic(self.network[-1][i].value)*\
                    logistic(1-self.network[-1][i].value)*\
                    (example[1][i]-result)

                    self.network[-1][i].error = error
                for i in range(len(self.network)-2,0,-1):
                    for j in range(len(self.network[i])):
                        error = logistic(self.network[i][j].value)*\
                        logistic(1-self.network[i][j].value)*\
                        self.network[i][j].sum_outgoing_weights()

                        self.network[i][j].error = error
                #update every weight in network using deltas
                for layer in self.network:
                    for node in layer:
                        for link in node.links:
                            link.weight = link.weight + .1 * node.value * node.error

        return self

    def forward_propagate(self, input_val):
        """ forward propagation one input matrix"""
        y_hat = []
        for i in range(len(input_val)):
            # propogate through input layer
            self.network[0][1].set_input(input_val[i])
            self.network[0][1].propogate()

        # activate and propogate through hidden layera
        h = 1
        while (h < (len(self.network) - 1)):
            for node in self.network[h]:
                node.activate()
                node.propogate()
                h += 1

        # activate through output layer, with single node
        self.network[len(self.network) - 1][0].activate()
        # chosen based on sigmoid function return value
        return self.predict_class()

def cross_validation(training, k, nn):
    """This function runs a cross-validation training/testing on k-fold dataset"""
    random.shuffle(training)
    folds = []
    for i in range(0, len(training), 10):
        fold = []
        for j in range(k):
            fold.append(training[i + j])
        folds.append(fold)

    errors = 0
    for fold in folds:
        #print(fold)
        folds_copy = copy.deepcopy(folds)
        folds_copy.remove(fold)
        training_set = itertools.chain.from_iterable(folds_copy)
        nn.back_propagation_learning(training_set)
        #print("hey")
        error = accuracy(nn, fold)
        #print(error)
        errors += error

    return "average accuracy: " + str(errors/len(folds))


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
    nn = NeuralNetwork([2, 6, 1])
    print(cross_validation(training, 10, nn))


if __name__ == "__main__":
    main()
