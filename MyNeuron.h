#ifndef _ETHY_NEURON_
#define _ETHY_NEURON_ 1

#include<bits/stdc++.h>
#include <stdexcept> // 用于抛出异常
#include <random> //随机数生成

typedef std::vector<double> my_vector;// to make it easier for programming.
/*
layer (h, i) to layer (h+1, j) will be stored in:
w[h][i][j]. w is an instance of my_power(3d-vector of double)
*/
typedef std::vector<my_vector> single_power;
typedef std::vector<single_power> my_power;

inline double getMSELoss(double& x1, double& x2);

class MyNeuron {
private:
	static int LR_VOKE;
	int epoches;//learning times
	//std::vector<std::vector<int>> layers;//todo: delete this.
	double learning;			//study rate
	my_power w;					//power, dimension is three
	//my_vector output;			//dinal output, dimension is one
	std::vector<my_vector> h;	//layer output storage; dimension is 2;
	std::vector<my_vector> o;	//after sigmoid output layer.
	std::vector<my_vector> b;	//bias, dimension is 2 to fix each layer h.
	bool isSameDouble(double d1, double d2);//
	void calculateOutput(my_vector& x, my_vector& y, single_power& power, my_vector& bias, my_vector& o_sigmoid);
	void train(my_vector& input, my_vector& labels);
	void backward(std::vector<my_vector>& data, my_vector& labels);

public:
	MyNeuron();
	MyNeuron(int epoches, double lr);
	MyNeuron(int epoches, double lr, std::vector<my_vector> h);
	//MyNeuron(int epoches, double lr, std::vector<std::)
	double sigmoid(double x);
	double d_sigmoid(double x);
	my_vector& forward(my_vector& data);
	//my_vector forward(std::vector<my_vector>& data);
	void train(std::vector<my_vector>& data, my_vector& label);
	void predict(std::vector<my_vector>& test_data, my_vector& test_label);
	my_vector& predict(my_vector& input);
	//my_vector predict(my_vector& input);
	bool predict(my_vector& input, double therehold);//todo

	//void printLoss();
};


#endif
