#ifndef MPL_H
#define MPL_H 1

#include "MyNeuron.h"
#include <vector>

class MPL {
private:
    MyNeuron neuron;

public:
    MPL();
    MPL(int epochs, double lr, int input_size, int layer_sizes[], int layer_length);
    double predict(std::vector<double>& data, double threshold);
    void train(std::vector<std::vector<double>>& inputs, std::vector<double>& labels);
    void setLR_VOKE(int voke);
};

#endif // MPL_H
