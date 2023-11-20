#ifndef _ETHY_NEURON_
#define _ETHY_NEURON_ 1

//#include<bits/stdc++.h>
#include<vector>
#include<iostream>
#include <cassert>
#include <stdexcept> // �����׳��쳣
#include <random> //���������

typedef std::vector<double> my_vector;// to make it easier for programming.
typedef std::vector<my_vector> single_power;
typedef std::vector<single_power> my_power;

/*
the power between layer (h, i) and layer (h+1, j) will be stored in:
w[h][i][j]. w is an instance of my_power(3d-vector of double)
*/

/*
* Ϊ���������Ԥ��ֵ����MSE��ʧ��
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
	�չ�������Ĭ��epoches=100, lr=0.01
	*/
	MyNeuron();
	/*
	Ĭ�ϲ�����һ��������磬�������룬һ�������
	*/
	MyNeuron(int epoches, double lr);
	/*
	@ params:
	epoches: ��������
	lr: ѧϰ��
	h: ���������롢����������layer��ȡ���ʱ����ȡh.back()[0]��Ϊ����ο���
	*/
	MyNeuron(int epoches, double lr, std::vector<my_vector> h);
	//MyNeuron(int epoches, double lr, std::vector<std::)

	/*
	@ params:
	epoches: ��������
	lr: ѧϰ��
	inputSize: ����ά��
	hiddenLayerSizes: �������ĵĳ��Ƚ��������ز�ĳ��ȣ���������е�ÿ��Ԫ�ؽ�����ÿ�����ز��ά�ȡ�
		��������{2,3,3}, ���ز��һ����Ԫ������2���ڶ���������Ԫ��������3.
	*/
	MyNeuron(int epoches, double lr, int inputSize, std::vector<int> hiddenLayerSizes);

	/*
	@ params:
	epoches: ��������
	lr: ѧϰ��
	inputSize: ����ά��
	hiddenLayerSizes: �������ĵĳ��Ƚ��������ز�ĳ��ȣ���������е�ÿ��Ԫ�ؽ�����ÿ�����ز��ά�ȡ�
		��������{2,3,3}, ���ز��һ����Ԫ������2���ڶ���������Ԫ��������3.
	hidenSize: ���ڲ���������һ�����飬������Ҫ����һ�����飨���ز��������ĳ��ȡ�
	@ comments
	���������Ҫ��������python�����ڷ�ֹC++��vector���ݺ�python��Ԫ��Ĵ��ݳ��ֵ����⡣
	*/
	MyNeuron(int epoches, double lr, int inputSize, int hiddenLayerSizes[], int hidenSize);

	/*
	�������
	*/
	virtual double sigmoid(double x);

	/*
	������ĵ�����
	*/
	virtual double d_sigmoid(double x);

	/*
	ǰ����㡣�������루һ��һά�������У���������ά�Ⱥ͹������趨������layer��ά�Ȳ�ͬ�����򽫻��˳���
	*/
	my_vector& forward(my_vector& data);
	//my_vector forward(std::vector<my_vector>& data);

	/*
	ѵ��������Ȩ��w��ƫ��b��
	�ú��������data��label����д�������
	@ params
	data: ��ά��m*n��ѵ�����顣m��ʾ���������� n��ʾÿ������ά���ġ�
		��Ҫע����ǣ�����ά��n��Ҫ��layer�ĵ�һ�㣨��h[0]����ά����ͬ�����������˳���
	label�����������Ĳο��������һ��һά�ĳ���Ϊm��double���顣
	*/
	void train(std::vector<my_vector>& data, my_vector& label);

	//deprecated
	void predict(std::vector<my_vector>& test_data, my_vector& test_label);

	/*
	Ԥ�������
	@ param
	input��һ��һά���顣������Ҫ��layer��һ�㣨h[0]���ĳ�����ͬ��������ͬ�����򽫻��˳���
	@ return
	���أ���������һ��layer����h.back()�������ã��⽫����layer���һ���һ����ֵ(h[h.size()-1][0]�����ı䡣
	������Ƽ����������
	*/
	my_vector& predict(my_vector& input);
	//my_vector predict(my_vector& input);

	/*
	Ԥ�������
	@ param
	input��һ��һά���顣������Ҫ��layer��һ�㣨h[0]���ĳ�����ͬ��������ͬ�����򽫻��˳���
	therehold: ��ֵ�����������򷵻��棨1.0�������򷵻ؼ٣�0.0����Ϊ�˱��ֿ���չ�ԣ�������ʱʹ��double���͡�
	@ return
	���أ���������һ��layer����h.back()���ĵ�һ��Ԫ�ء�ͨ������ֻ��һ������������������������ǿ����õġ�
	*/
	double predict(my_vector& input, double therehold);//todo

	//void printLoss();
	void setLR_VOKE(int LR_VOKE);
};


#endif
