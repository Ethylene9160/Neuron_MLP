#include <random>
#include <cstdlib> // for std::srand
#include<ctime>

#include"MyNeuron.h"
#include"MPL.h"


bool judgeOther(double x, double y) {
    //区域：x和y处在x^2+y^2<0.25 or (x-1)^2+(y-1)^2<0.25的区域内
    return (x * x + y * y < 0.25) || (x - 1) * (x - 1) < 0.25;
}

int main() {
    //std::srand(static_cast<unsigned>(time(0)));
    //printf("hello\n");
    // 创建训练数据
    // 使用随机设备作为种子
    //std::random_device rd;
    // 使用 Mersenne Twister 引擎
    //std::mt19937 gen(rd());
    // 定义随机数分布，在 [0.0, 1.0) 范围内生成浮点数
    //std::uniform_real_distribution<> dis(0.0, 1.0);

    int train_size = 600;
    std::vector<my_vector> trainData(train_size);
    my_vector trainLabel(train_size);
    for (int i = 0; i < train_size; ++i) {
        double x = (double)rand() / RAND_MAX;  // 随机生成 0 到 1 之间的数
        double y = (double)rand() / RAND_MAX;
        //double x = dis(gen)/2;
        //double y = dis(gen)/2;
        trainData[i] = { x, y };
        //trainLabel[i] = (x + y > 0.5 ? 1.0 : 0.0);  // 简单的线性可分规则
        //trainLabel[i] = (x*x + y*y > 1 ? 1.0 : 0.0);  // 圆
        trainLabel[i] = judgeOther(x, y) ? 1.0 : 0.0;
    }
    std::vector<my_vector> testData(500);
    my_vector testLabel(500);

    for (int i = 0; i < 500; ++i) {
        double x = (double)rand() / RAND_MAX / 2;  // 随机生成 0 到 1 之间的数
        double y = (double)rand() / RAND_MAX / 2;
        //double x = dis(gen);
        //double y = dis(gen);
        testData[i] = { x, y };
        //testLabel[i] = (x + y > 0.5 ? 1.0 : 0.0);  // 简单的线性可分规则
        //testLabel[i] = (x * x + y * y > 1 ? 1.0 : 0.0);//圆外为1，圆内为0
        testLabel[i] = judgeOther(x, y) ? 1.0 : 0.0;
    }
    printf("The first step is over!\n");

    // 创建神经网络实例
    //MyNeuron neuron(2000, 0.025, { {0,0}, {0,0, 0, 0,0,0,0,0,0}, {0} });  // 假设有 100 个 epoch 和 0.008 的学习率
    int hiddenLayer[] = { 9 };
    MyNeuron neuron(2000, 0.025, 2, hiddenLayer, sizeof(hiddenLayer)/sizeof(hiddenLayer[0]));
    MPL mpl(2000, 0.025, 2, hiddenLayer, sizeof(hiddenLayer) / sizeof(hiddenLayer[0]));
    printf("the neuron has been newed!\n");
    // 训练网络
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
    printf("\ncorrect rate is：%.1f\n", (double)corSum / 5.0);
    // 测试网络
    // ... 这里可以添加一些测试代码来验证网络的表现
    char xixixi = 0;
    std::cin >> xixixi;
    return 0;
}