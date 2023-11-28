#include <random>
#include <cstdlib> // for std::srand
#include <ctime>

#include"MyNeuron.h"
#include"MPL.h"

#include "FileController.h"

class ReLuMLP:public MyNeuron {


    double sigmoid(double x) override {
        //return x > 0.0 ? x : 0.0;
        return x > 0.0 ? x : 0.01 * x;
    }

    double d_sigmoid(double x) override {
        //return x > 0.0 ? 1.0 : 0.0;
        return x > 0.0 ? 1.0 : 0.01;
    }
public:
    ReLuMLP(int ep, double lr, int is, std::vector<int>& layers) :MyNeuron(ep, lr, is, layers) {}
};


class OrthMLP :public MyNeuron{

    void orth(my_vector& self_h, my_vector& no_use_array) override {
        double mu = 0.0, delta_2 = 0.0;
        int length = self_h.size();
        for (size_t i = 0; i < length; i++)
        {
            //batch_output[i] = self_h[i];
            mu += self_h[i];
        }

        mu /= (double)length;
        for (double& p : self_h) {
            double val = p - mu;
            delta_2 += val * val;
        }
        delta_2 /= (double)length;
        //printf("%f\n",delta_2);
        static const double epsilon = 0.0001;
        double fenmu = (delta_2 > epsilon) ? sqrt(delta_2) : epsilon;
        //double fenmu = sqrt(delta_2);
        if (0) {
            for (size_t i = 0; i < length; ++i)
            {
                //batch_output[i] = self_h[i];
                //batch_output[i] = self_h[i];
            }
        }
        else {
            for (size_t i = 0; i < length; ++i)
            {
                //batch_output[i] = (self_h[i] - mu) / fenmu;
                self_h[i] = (self_h[i] - mu) / fenmu;
                //batch_output[i] = self_h[i];
            }
        }
    }
public:
    OrthMLP(int ep, double lr, int is, std::vector<int>& layers) :MyNeuron(ep, lr, is, layers) {}
};

bool judgeOther(double x, double y) {
    //区域：x和y处在x^2+y^2<0.25 or (x-1)^2+(y-1)^2<0.25的区域内
    return (x * x + y * y < 0.25) || (x - 1) * (x - 1)+(y-1)*(y-1) < 0.25;
}

void resolveParams(double& d) {
    if (d > 0.0) return;
    d = 0.0;
}

int main() {
    std::srand(static_cast<unsigned>(time(0)));
    //printf("hello\n");
    // 创建训练数据
    // 使用随机设备作为种子
    std::random_device rd;
    // 使用 Mersenne Twister 引擎
    std::mt19937 gen(rd());
    // 定义随机数分布，在 [0.0, 1.0) 范围内生成浮点数
    std::uniform_real_distribution<> dis(0.0, 1.0);

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

    //写入数据到本地
    //saveDataToFile(trainData, trainLabel, "data.txt");

    std::vector<my_vector> testData(500);
    my_vector testLabel(500);

    for (int i = 0; i < 500; ++i) {
        //double x = (double)rand() / RAND_MAX / 2;  // 随机生成 0 到 1 之间的数
        //double y = (double)rand() / RAND_MAX / 2;
        double x = dis(gen);
        double y = dis(gen);
        testData[i] = { x, y };
        //testLabel[i] = (x + y > 0.5 ? 1.0 : 0.0);  // 简单的线性可分规则
        //testLabel[i] = (x * x + y * y > 1 ? 1.0 : 0.0);//圆外为1，圆内为0
        testLabel[i] = judgeOther(x, y) ? 1.0 : 0.0;
    }
    printf("The first step is over!\n");

    // 创建神经网络实例
    //MyNeuron neuron(2000, 0.025, { {0,0}, {0,0, 0, 0,0,0,0,0,0}, {0} });  // 假设有 100 个 epoch 和 0.008 的学习率
    //int hiddenLayer[] = { 3,5 };
    //MyNeuron neuron(2000, 0.025, 2, hiddenLayer, sizeof(hiddenLayer)/sizeof(hiddenLayer[0]));
    //MPL mpl(2000, 0.025, 2, hiddenLayer, sizeof(hiddenLayer) / sizeof(hiddenLayer[0]));
    //printf("the neuron has been newed!\n");
    // 训练网络
    //neuron.train(trainData, trainLabel);
    //mpl.train(trainData, trainLabel);

    //std::vector<int> layers = { 3,5,4,6,7,8,9};
    std::vector<int> layers = { 3,4 };
    OrthMLP sigmoidNeuron(2000, 0.025, 2, layers);
    sigmoidNeuron.setLambda(0.5);
    ReLuMLP reLuNeuron(8000, 0.0005, 2, layers);
    printf("some test data compare\n");
    sigmoidNeuron.setLR_VOKE(1000);
    reLuNeuron.setLR_VOKE(400);

    //读取data
    //loadDataFromFile(trainData, trainLabel, "data.txt");
    //reLuNeuron.train(trainData, trainLabel);
    sigmoidNeuron.train(trainData, trainLabel);
    /*
    for (int i = 0; i < 5; ++i)
    {
        my_vector& outputs = neuron.predict(testData[i]);
        printf("input is: %f and %f\t\n", testData[i][0], testData[i][1]);
        printf("label is: %f, and prediction is %f\n", testLabel[i], outputs[0]);
    }
    */

    //int corSum = 0;
    int sigmoidSum = 0, reLuSum = 0;


    double sigmoid_recall = 0.0, sigmoid_precision = 0.0, sigmoidF1 = 0.0,
        relu_recall = 0.0, relu_precision = 0.0, reluF1 = 0.0;

    double sigmoid_f1 = 0.0, relu_f1 = 0.0;

    int sigmoidTP = 0.0, sigmoidFN = 0.0, sigmoidFP = 0.0, sigmoidTN = 0.0,
        reluTP = 0.0, reluFN = 0.0, reluFP = 0.0, reluTN = 0.0;

    for (int i = 0; i < 500; ++i) {
        //my_vector& out = neuron.predict(testData[i]);
        double sigmoid_pred = sigmoidNeuron.predict(testData[i], 0.5),
            relu_pred = reLuNeuron.predict(testData[i], 0.5);

        if (testLabel[i] > 0.5) {
            if (sigmoid_pred > 0.5) {
                sigmoidTP+=1;
            }
            else {
                sigmoidFN+=1;
            }
            if (relu_pred > 0.5) {
                reluTP+=1;
            }
            else {
                reluFN+=1;
            }
        }
        else {
            if (sigmoid_pred > 0.5) {
                sigmoidFP+=1;
            }
            else {
                //sigmoidSum += 1;
                sigmoidTN += 1;
            }
            if (relu_pred > 0.5) {
                reluFP+=1;
            }
            else {
                //reLuSum += 1;
                reluTN += 1;
            }
        }

        
        //if (sigmoid_pred == testLabel[i]) sigmoidSum += 1;
        //if (relu_pred == testLabel[i]) reLuSum += 1;
        //if (out[0] == testLabel[i]) corSum += 1;
        //if (mpl.predict(testData[i], 0.5) == testLabel[i]) corSum += 1;
    }
    /*
    my_vector v = { {0.05}, {0.14} };
    my_vector& outputs = neuron.predict(v);
    printf("\n%f\n", outputs[0]);
    */
    //printf("\ncorrect rate is：%.1f\n", (double)corSum / 5.0);
    sigmoidSum = sigmoidTP+sigmoidTN;
    reLuSum =reluTP+reluTN;
    sigmoid_recall = double(sigmoidTP) / double(sigmoidTP + sigmoidFN);
    sigmoid_precision = double(sigmoidTP) / double(sigmoidTP + sigmoidFP);
    resolveParams(sigmoid_recall);
    resolveParams(sigmoid_precision);
    sigmoidF1 = (2.0 * sigmoid_precision * sigmoid_recall) / (sigmoid_precision + sigmoid_recall);
    resolveParams(sigmoidF1);
    relu_recall = double(reluTP) / double(reluTP + reluFN);
    relu_precision = double(reluTP) / double(reluTP + reluFP);

    resolveParams(relu_recall);
    resolveParams(relu_precision);
    reluF1 = (2.0 * relu_precision * relu_recall) / (relu_precision + relu_recall);
    resolveParams(reluF1);

    

    printf("sigmoidTP:%d\n", sigmoidTP);
    printf("sigmoidFN:%d\n", sigmoidFN);
    printf("sigmoidFP:%d\n", sigmoidFP);
    printf("sigmoidTN:%d\n", sigmoidTN);
    printf("sigmoid SUM:%d\n", sigmoidSum);


    printf("sigmoid;\n");
    printf("recall: %f\n", sigmoid_recall);
    printf("precision: %f\n", sigmoid_precision);
    printf("f1 score: %f\n", sigmoidF1);
    printf("correct rate of sigmoid is:%.1f\n", (double)sigmoidSum / 5.0);
    printf("\n");
    printf("relu;\n");
    printf("recall: %f\n", relu_recall);
    printf("precision: %f\n", relu_precision);
    printf("f1 score: %f\n", reluF1);
    printf("\ncorrect rate of ReLu is:%.1f\n", (double)reLuSum / 5.0);
    // 测试网络
    // ... 这里可以添加一些测试代码来验证网络的表现
    char xixixi = 0;
    std::cin >> xixixi;
    return 0;
}