#include "MPL.h"

MPL::MPL() {
    neuron = MyNeuron();
}

MPL::MPL(int epochs, double lr, int input_size, int layer_sizes[], int layer_length) {
    neuron = MyNeuron(epochs, lr, input_size, layer_sizes, layer_length);
}

double MPL::predict(std::vector<double>& data, double threshold) {
    return neuron.predict(data, threshold);
}

void MPL::train(std::vector<std::vector<double>>& inputs, std::vector<double>& labels) {
    neuron.train(inputs, labels);
}

void MPL::setLR_VOKE(int voke) {
    neuron.setLR_VOKE(voke);
}
