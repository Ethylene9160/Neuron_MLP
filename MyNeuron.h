#ifndef _ETHY_NEURON_
#define _ETHY_NEURON_ 1

//#include<bits/stdc++.h>
#include<vector>
#include<iostream>
#include <cassert>
#include <stdexcept> // 用于抛出异常
#include <random> //随机数生成

typedef std::vector<double> my_vector;// to make it easier for programming.
typedef std::vector<my_vector> single_power;
typedef std::vector<single_power> my_power;

/*
the power between layer (h, i) and layer (h+1, j) will be stored in:
w[h][i][j]. w is an instance of my_power(3d-vector of double)
*/

/*
* 为单个输出和预测值计算MSE损失。
calculate MSE loss for single output and single prediction.
*/
inline double getMSELoss(double x1, double x2);

class MyNeuron {
private:
	int LR_VOKE;				//after LR_VOKE epoches, the prog will print the loss value , current w and b.
	int epoches;				//learning times
	double learning;			//study rate
	my_power w;					//power, dimension is three
	//my_vector output;			//dinal output, dimension is one
	std::vector<my_vector> h;	//layer output storage; dimension is 2;
	std::vector<my_vector> o;	//after sigmoid output layer.
	std::vector<my_vector> b;	//bias, dimension is 2 to fix each layer h.

	void init(int epoches, double lr, std::vector<my_vector> h);

	bool isSameDouble(double d1, double d2);

	void calculateOutput(my_vector& x, my_vector& y, single_power& power, my_vector& bias, my_vector& o_sigmoid);
	
	//deprecated
	void train(my_vector& input, my_vector& labels);

	//deprecated
	void backward(std::vector<my_vector>& data, my_vector& labels);

	virtual void orth(my_vector& data);

public:
	/*
	空构造器。默认epoches=100, lr=0.01
	*/
	MyNeuron();
	/*
	默认层数是一层的神经网络，两个输入，一个输出。
	*/
	MyNeuron(int epoches, double lr);
	/*
	@ params:
	epoches: 迭代次数
	lr: 学习率
	h: 包含了输入、输出层的所有layer。取输出时，仅取h.back()[0]作为输出参考。
	*/
	MyNeuron(int epoches, double lr, std::vector<my_vector> h);
	//MyNeuron(int epoches, double lr, std::vector<std::)

	/*
	@ params:
	epoches: 迭代次数
	lr: 学习率
	inputSize: 输入维度
	hiddenLayerSizes: 这个数组的的长度将会是隐藏层的长度，这个数组中的每个元素将会是每个隐藏层的维度。
		例如输入{2,3,3}, 隐藏层第一层神经元数量是2，第二、三层神经元数量都是3.
	*/
	MyNeuron(int epoches, double lr, int inputSize, std::vector<int> hiddenLayerSizes);

	/*
	@ params:
	epoches: 迭代次数
	lr: 学习率
	inputSize: 输入维度
	hiddenLayerSizes: 这个数组的的长度将会是隐藏层的长度，这个数组中的每个元素将会是每个隐藏层的维度。
		例如输入{2,3,3}, 隐藏层第一层神经元数量是2，第二、三层神经元数量都是3.
	hidenSize: 由于参数传递是一个数组，所以需要传入一个数组（隐藏层数量）的长度。
	@ comments
	这个函数主要用于适配python，用于防止C++中vector传递和python中元组的传递出现的问题。
	*/
	MyNeuron(int epoches, double lr, int inputSize, int hiddenLayerSizes[], int hidenSize);

	/*
	激活函数。
	*/
	virtual double sigmoid(double x);

	/*
	激活函数的导函数
	*/
	virtual double d_sigmoid(double x);

	/*
	前向计算。放入输入（一个一维浮点序列）。若输入维度和构造器设定的输入layer的维度不同，程序将会退出。
	*/
	my_vector& forward(my_vector& data);
	//my_vector forward(std::vector<my_vector>& data);

	/*
	训练。计算权重w和偏置b。
	该函数不会对data和label进行写入操作。
	@ params
	data: 二维（m*n）训练数组。m表示样本数量， n表示每个输入维数的。
		需要注意的是，输入维数n需要和layer的第一层（即h[0]）的维度相同。否则程序会退出。
	label：测试样本的参考输出。是一个一维的长度为m的double数组。
	*/
	void train(std::vector<my_vector>& data, my_vector& label);

	//deprecated
	void predict(std::vector<my_vector>& test_data, my_vector& test_label);

	/*
	预测输出。
	@ param
	input：一个一维数组。长度需要与layer第一层（h[0]）的长度相同。倘若不同，程序将会退出。
	@ return
	返回：输出（最后一层layer，即h.back()）的引用（这将导致layer最后一层第一个数值(h[h.size()-1][0]）被改变。
	因而不推荐这个函数。
	*/
	my_vector& predict(my_vector& input);
	//my_vector predict(my_vector& input);

	/*
	预测输出。
	@ param
	input：一个一维数组。长度需要与layer第一层（h[0]）的长度相同。倘若不同，程序将会退出。
	therehold: 阈值。超过它，则返回真（1.0），否则返回假（0.0）。为了保持可拓展性，返回暂时使用double类型。
	@ return
	返回：输出（最后一层layer，即h.back()）的第一个元素。通常对于只有一个输出的神经网络来讲，这样是可以用的。
	*/
	double predict(my_vector& input, double therehold);//todo

	//void printLoss();
	void setLR_VOKE(int LR_VOKE);
};


#endif
