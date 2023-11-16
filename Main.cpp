#include <random>
#include <cstdlib> // for std::srand
#include<ctime>

#include"MyNeuron.h"
#include"MPL.h"


bool judgeOther(double x, double y) {
    //����x��y����x^2+y^2<0.25 or (x-1)^2+(y-1)^2<0.25��������
    return (x * x + y * y < 0.25) || (x - 1) * (x - 1) < 0.25;
}

int main() {
    //std::srand(static_cast<unsigned>(time(0)));
    //printf("hello\n");
    // ����ѵ������
    // ʹ������豸��Ϊ����
    //std::random_device rd;
    // ʹ�� Mersenne Twister ����
    //std::mt19937 gen(rd());
    // ����������ֲ����� [0.0, 1.0) ��Χ�����ɸ�����
    //std::uniform_real_distribution<> dis(0.0, 1.0);

    int train_size = 600;
    std::vector<my_vector> trainData(train_size);
    my_vector trainLabel(train_size);
    for (int i = 0; i < train_size; ++i) {
        double x = (double)rand() / RAND_MAX;  // ������� 0 �� 1 ֮�����
        double y = (double)rand() / RAND_MAX;
        //double x = dis(gen)/2;
        //double y = dis(gen)/2;
        trainData[i] = { x, y };
        //trainLabel[i] = (x + y > 0.5 ? 1.0 : 0.0);  // �򵥵����Կɷֹ���
        //trainLabel[i] = (x*x + y*y > 1 ? 1.0 : 0.0);  // Բ
        trainLabel[i] = judgeOther(x, y) ? 1.0 : 0.0;
    }
    std::vector<my_vector> testData(500);
    my_vector testLabel(500);

    for (int i = 0; i < 500; ++i) {
        double x = (double)rand() / RAND_MAX / 2;  // ������� 0 �� 1 ֮�����
        double y = (double)rand() / RAND_MAX / 2;
        //double x = dis(gen);
        //double y = dis(gen);
        testData[i] = { x, y };
        //testLabel[i] = (x + y > 0.5 ? 1.0 : 0.0);  // �򵥵����Կɷֹ���
        //testLabel[i] = (x * x + y * y > 1 ? 1.0 : 0.0);//Բ��Ϊ1��Բ��Ϊ0
        testLabel[i] = judgeOther(x, y) ? 1.0 : 0.0;
    }
    printf("The first step is over!\n");

    // ����������ʵ��
    //MyNeuron neuron(2000, 0.025, { {0,0}, {0,0, 0, 0,0,0,0,0,0}, {0} });  // ������ 100 �� epoch �� 0.008 ��ѧϰ��
    int hiddenLayer[] = { 9 };
    MyNeuron neuron(2000, 0.025, 2, hiddenLayer, sizeof(hiddenLayer)/sizeof(hiddenLayer[0]));
    MPL mpl(2000, 0.025, 2, hiddenLayer, sizeof(hiddenLayer) / sizeof(hiddenLayer[0]));
    printf("the neuron has been newed!\n");
    // ѵ������
    //neuron.train(trainData, trainLabel);
    mpl.train(trainData, trainLabel);
    printf("some test data compare\n");
    /*
    for (int i = 0; i < 5; ++i)
    {
        my_vector& outputs = neuron.predict(testData[i]);
        printf("input is: %f and %f\t\n", testData[i][0], testData[i][1]);
        printf("label is: %f, and prediction is %f\n", testLabel[i], outputs[0]);
    }
    */

    int corSum = 0;
    for (int i = 0; i < 500; ++i) {
        //my_vector& out = neuron.predict(testData[i]);

        //if (out[0] == testLabel[i]) corSum += 1;
        if (mpl.predict(testData[i], 0.5) == testLabel[i]) corSum += 1;
    }
    /*
    my_vector v = { {0.05}, {0.14} };
    my_vector& outputs = neuron.predict(v);
    printf("\n%f\n", outputs[0]);
    */
    printf("\ncorrect rate is��%.1f\n", (double)corSum / 5.0);
    // ��������
    // ... ����������һЩ���Դ�������֤����ı���
    char xixixi = 0;
    std::cin >> xixixi;
    return 0;
}