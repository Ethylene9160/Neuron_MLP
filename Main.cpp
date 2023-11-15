#include<bits/stdc++.h>
#include <random>

#include"MyNeuron.h"

int main() {
    std::srand(static_cast<unsigned>(std::time(0)));
    //printf("hello\n");
    // ����ѵ������
    // ʹ������豸��Ϊ����
    std::random_device rd;
    // ʹ�� Mersenne Twister ����
    std::mt19937 gen(rd());
    // ����������ֲ����� [0.0, 1.0) ��Χ�����ɸ�����
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int train_size = 10000;
    std::vector<my_vector> trainData(train_size);
    my_vector trainLabel(train_size);
    for (int i = 0; i < train_size; ++i) {
        double x = (double)rand() / RAND_MAX / 2;  // ������� 0 �� 1 ֮�����
        double y = (double)rand() / RAND_MAX / 2;
        //double x = dis(gen)/2;
        //double y = dis(gen)/2;
        trainData[i] = { x, y };
        trainLabel[i] = (x + y > 0.5 ? 1.0 : 0.0);  // �򵥵����Կɷֹ���
    }
    std::vector<my_vector> testData(500);
    my_vector testLabel(500);

    for (int i = 0; i < 500; ++i) {
        //double x = (double)rand() / RAND_MAX / 2;  // ������� 0 �� 1 ֮�����
        //double y = (double)rand() / RAND_MAX / 2;
        double x = dis(gen) / 2;
        double y = dis(gen) / 2;
        testData[i] = { x, y };
        testLabel[i] = (x + y > 0.5 ? 1.0 : 0.0);  // �򵥵����Կɷֹ���
    }
    printf("The first step is over!\n");

    // ����������ʵ��
    MyNeuron neuron(100, 0.008, { {0,0}, {0,0}, {0} });  // ������ 100 �� epoch �� 0.008 ��ѧϰ��
    printf("the neuron has been newed!\n");
    // ѵ������
    neuron.train(trainData, trainLabel);
    printf("some test data compare\n");
    for (int i = 0; i < 5; ++i)
    {
        my_vector& outputs = neuron.predict(testData[i]);
        printf("input is: %f and %f\t\n", testData[i][0], testData[i][1]);
        printf("label is: %f, and prediction is %f\n", testLabel[i], outputs[0]);
    }

    int corSum = 0;
    for (int i = 0; i < 500; ++i) {
        my_vector& out = neuron.predict(testData[i]);
        if (out[0] == testLabel[i]) corSum += 1;
    }

    my_vector v = { {0.05}, {0.14} };
    my_vector& outputs = neuron.predict(v);
    printf("\n%f\n", outputs[0]);
    printf("\ncorrect rate is��%f\n", (double)corSum / 500.0);
    // ��������
    // ... ����������һЩ���Դ�������֤����ı���
    char xixixi = 0;
    std::cin >> xixixi;
    return 0;
}