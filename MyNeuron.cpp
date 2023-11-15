#include "MyNeuron.h"
template<typename T>

//deprecated
void printArray(int dimension, std::vector<T>& arr) {
    return;
    /*
    if (dimension == 1) {
        for (T at : arr) {
            std::cout << at << "\t";
        }
        std::cout << std::endl;
    }
    else if (dimension > 1) {
        for(auto&a:arr)
            printArray(dimension - 1, a);
    }
    */
}

inline double getMSELoss(double& x1, double& x2)
{
    double d = x1 - x2;
    return d * d;
}

bool MyNeuron::isSameDouble(double d1, double d2)
{
    return d1 == d2;
}

void MyNeuron::calculateOutput(my_vector& x, my_vector& y, single_power& power, my_vector& bias, my_vector& o_sigmoid)
{
    // 确保权重矩阵的列数与输出向量的尺寸相匹配
    for (size_t i = 0; i < power.size(); ++i) {
        assert(power[i].size() == y.size());
    }

    // 确保偏置向量的尺寸与输出向量的尺寸相匹配
    assert(bias.size() == y.size());

    // 计算输出向量的每个元素
    for (int j = 0; j < y.size(); ++j) {
        y[j] = 0.0;
        for (int k = 0; k < x.size(); k++) {
            // 确保权重矩阵的行数与输入向量的尺寸相匹配
            assert(k < power.size());
            // 确保当前的k行有一个对应的j列
            assert(j < power[k].size());

            y[j] += x[k] * power[k][j];
        }
        //更新y
        y[j] = (y[j] + bias[j]);
        // 应用激活函数
        o_sigmoid[j] = this->sigmoid(y[j]);
    }
    /*
    for (int j = 0; j < y.size(); ++j) {
        y[j] = 0.0;
        for (int k = 0; k < x.size(); k++) {
            y[j] += x[k] * power[k][j];
        }
        y[j] = this->sigmoid(y[j]+bias[j]);
    }*/
}

void MyNeuron::train(my_vector& input, my_vector& labels)
{
    //deprecated
    return;
}

//deprecated
void MyNeuron::backward(std::vector<my_vector>& data, my_vector& labels)
{
    int data_size = data.size();
    my_vector predictions(data_size);
    my_vector output_errors(data_size);
    my_vector output_deltas(data_size);
    //my_vector outputs(data_size);
    //更新my_vector output_errors
    for (int i = 0; i < data_size; ++i) {
        predictions[i] = forward(data[i])[0];
        output_errors[i] = predictions[i] - labels[i];
        output_deltas[i] = output_errors[i] * this->d_sigmoid(predictions[i]);
    }

    //update all msgs.
    //h 包括了一层输入层，一层输出层，主要是方便归一计算。
    int h_size = h.size();
    std::vector<my_vector> all_errors(h_size);
    std::vector<my_vector> all_deltas(h_size);
    //all_errors[h_size-1]
    //接下来，需要更新每一个h。这样方便接下来的反向计算。
    //由于forward函数中，每一次更新都会使h改变为新的h并保存下来，那么：
    //由于上面已经调用了一次forward， 此时的h已经被成功更新了。
    //那么：

    //初始化all_errors, all_deltas:
    for (int i = 0; i < h_size; ++i) {
        all_errors[i] = my_vector(h[i].size());
        all_deltas[i] = my_vector(h[i].size());
    }

    //将输出层的值赋值给它
    all_errors[all_errors.size() - 1].assign(output_errors.begin(), output_errors.end());
    all_deltas[all_deltas.size() - 1].assign(output_deltas.begin(), output_deltas.end());

    //
    for (int layerIndex = all_errors.size() - 1; layerIndex > 0; --layerIndex) {
        for (int neuronIndex = 0; neuronIndex < 0; ++neuronIndex) {//todo
            all_errors[layerIndex - 1][neuronIndex] = all_deltas[layerIndex][neuronIndex] * d_sigmoid(h[layerIndex][neuronIndex]);
        }

        //all_errors[i - 1] = all_deltas[i] * d_sigmoid(predictions[i]);
        //all_deltas[i - 1] = all_errors[i - 1] * d_sigmoid();
    }

}

MyNeuron::MyNeuron() :MyNeuron(100, 0.01)
{
}


MyNeuron::MyNeuron(int eopches, double lr) :
    MyNeuron(eopches, lr, { {0,0},{0} })
{
}

MyNeuron::MyNeuron(int eopches, double lr, std::vector<my_vector> h)
{
    this->epoches = eopches;
    this->learning = lr;
    //this->layers = layers;
    this->h = h;
    this->o = h;
    this->b = std::vector<my_vector>(h.size());
    printf("size of h is: \n");
    for (int i = 0; i < h.size(); ++i) {
        printf("size%d is %d\n", i, h[i].size());
    }
    printf("\n");
    printf("size of b is: \n");
    for (int i = 0; i < h.size(); ++i) {
        b[i] = my_vector(h[i].size(), 0);
        printf("size%d is %d\n", i, b[i].size());
    }


    this->w = my_power(this->h.size() - 1);
    for (int i = 0; i < this->w.size(); ++i) {
        this->w[i] = single_power(h[i].size(), my_vector(h[i + 1].size()));
    }
    // 随机赋予w值
    // 初始化随机数生成器
    std::default_random_engine generator(static_cast<unsigned>(time(0)));
    std::uniform_real_distribution<double> distribution(-1.0, 1.0); // 例如在-1.0到1.0之间生成随机数

    // 初始化权重矩阵
    this->w = my_power(this->h.size() - 1);
    for (int i = 0; i < this->w.size(); ++i) {
        this->w[i] = single_power(h[i].size(), my_vector(h[i + 1].size()));
        for (int j = 0; j < w[i].size(); ++j) {
            for (int k = 0; k < w[i][j].size(); ++k) {
                w[i][j][k] = distribution(generator); // 使用随机数填充
            }
        }
    }


    printf("\nsize of w is: \n");
    for (int i = 0; i < w.size(); ++i) {
        printf("size of %dth of w is %d * %d\n", i, w[i].size(), w[i][0].size());
    }

    printf("================\n init w will be: \n");
    for (int i = 0; i < w.size(); ++i) {
        printf("w%d will be:\n", i);
        for (int j = 0; j < w[i].size(); ++j) {
            for (int k = 0; k < w[i][j].size(); ++k) {
                printf("%f\t", w[i][j][k]);
            }
            printf("\n");
        }
    }
    printf("================\n init b will be: \n");
    for (int i = 0; i < b.size(); ++i) {
        for (int j = 0; j < b[i].size(); ++j) {
            printf("b%d%d is: %f\t", i, j, b[i][j]);
        }
        printf("\n");
    }

}

double MyNeuron::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double MyNeuron::d_sigmoid(double x)
{
    // the derivaty of 1/(1+exp(-x)) will be:
    // exp(-x)(1+exp(-x))^-2
    //=1/(1+exp(-x)) times exp(-x)/(1+exp(-x))
    // to lower the time complexity, use y.
    double y = sigmoid(x);
    return y * (1 - y);
}
/*
my_vector&MyNeuron::forward(my_vector& data)
{
    // forward!! todo: 未考虑偏置。

    //int size = data.size();
    //if (size != this->layers.size()) return 0.0;
    //double sum = 0.0;
    //for (int i = 0; i < size; i++) {
    //    sum += this->layers[layer][i] * data[i];
    //}

    //input 作为 h[0], h[size-1]作为 output， 方便写了。
    int data_size = data.size();
    if (data_size != h[0].size()) return this->h[this->h.size() - 1];
    this->h[0].assign(data.begin(), data.end());
    int i_max = this->h.size() - 1;
    for (int i = 0; i < i_max; i++) {

        //int j_max = this->h[i+1].size();
        //for (int j = 0; j < j_max; j++) {
        //    for (int k = 0; k < h[i].size(); ++k) {
        //        this->h[i + 1][j] += this->h[i][k] * this->w[i][k][j];
        //    }
        //    this->h[i + 1][j] = this->sigmoid(this->h[i + 1][j]);//access sigmoid function.
        //}

        this->calculateOutput(h[i], h[i + 1], w[i], b[i+1]);
    }

    return this->h[this->h.size()-1];

}
*/
my_vector& MyNeuron::forward(my_vector& data) {
    // 确保输入数据的尺寸与网络输入层的尺寸相匹配
    if (data.size() != h[0].size()) {
        throw std::invalid_argument("Size of input data does not match the size of the network's input layer.");
    }

    // 用输入数据初始化第一层的输出
    h[0].assign(data.begin(), data.end());
    o[0].assign(data.begin(), data.end());//输入层，不需要sigmoid
    // 进行前向传播
    int i_max = this->h.size() - 1;
    for (int i = 0; i < i_max; i++) {
        // 在进行矩阵乘法之前，确保索引有效
        if (i >= w.size() || i + 1 >= h.size() || i + 1 >= b.size()) {
            throw std::out_of_range("Index out of range during forward pass.");
        }

        // 检查权重矩阵的维度是否正确
        if (h[i].size() != w[i].size()) {
            throw std::invalid_argument("Mismatch between layer output size and weight matrix size.");
        }

        // 检查权重矩阵的每个向量的尺寸是否与下一层的尺寸匹配
        for (size_t k = 0; k < w[i].size(); ++k) {
            if (w[i][k].size() != h[i + 1].size()) {
                throw std::invalid_argument("Mismatch between weight matrix size and next layer size.");
            }
        }

        // 计算下一层的输出
        //this->calculateOutput(h[i], h[i + 1], w[i], b[i + 1]);
        this->calculateOutput(o[i], h[i + 1], w[i], b[i + 1], o[i + 1]);
        //printf("new h will be:\n");
        //for (int ti = 0; ti < h[i+1].size(); ++ti) {

            //printf("h%d%d is: %f\t", i+1, ti, h[i+1][ti]);
        //}
    }

    // 返回最后一层的输出
    //return this->h[this->h.size() - 1];
    return this->o[this->o.size() - 1];
}


/*

void MyNeuron::train(std::vector<my_vector>& data, my_vector& label)
{
    //假设最终的输出只有一个维度。即h[size-1].size() = 1.
    //不然的话 能力实在暂时不够。
    for (int epc = 0; epc < epoches; ++epc) {
        int data_size = data.size();
        std::vector<my_vector> h_sum;
        h_sum.assign(h.begin(), h.end());
        //for (int i = 0; i < h.size(); ++i) {
            //h_sum[i].assign(h[i].begin(), h[i].end());
        //}
        for (int i = 0; i < data_size; ++i) {
            //train each data.

            my_vector&x = data[i];
            h_sum[0].assign(x.begin(), x.end());
            //calculateOutput(x, NULL, this->w[]);

            //计算h们
            int hi_max = h_sum.size() - 1;
            for (int hi = 0; hi < hi_max; ++hi) {
                this->calculateOutput(h[hi], h[hi + 1], w[hi], b[hi + 1]);
            }

            double pred = h_sum[hi_max][0];//假设输出是1*1！！
            double d_loss_pred = -2 * (label[i] - pred);
            double loss_constant = d_loss_pred * learning;//降低运算复杂度

            my_power d_pred_w;
            d_pred_w.assign(w.begin(), w.end());

            for (int i = 0; i < h.size() - 1; ++i) {
                for (int j = 0; j < h[i+1].size(); ++j) {
                    for (int k = 0; k < h[i].size(); ++k) {
                        d_pred_w[i][k][j] = h_sum[i][j] * d_sigmoid(h[i + 1][j]);
                        w[i][k][j] -= loss_constant*;//todo
                    }
                    b[i][j] -= 0.0;//todo
                }
            }
        }

        if (epc % 10 == 0) {
            double loss = 0;
            for (int i = 0; i < data_size; ++i) {
                my_vector&output = forward(data[i]);
                //假设只有一个是输出。
                double pred = output[0];
                loss += getMSELoss(pred, label[i]);
            }

        }

    }

    //todo!!
}
*/

void MyNeuron::train(std::vector<my_vector>& data, my_vector& label) {
    assert(data.size() == label.size());  // 确保数据和标签的数量匹配
    //假定输出维度为1*1.多维输出能力不够，不会。
    for (int epoch = 0; epoch < epoches; ++epoch) {
        for (size_t dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
            assert(dataIndex < label.size());  // 确保标签索引在范围内

            //printf("train-forward\n");
            // 前向传播
            my_vector output = forward(data[dataIndex]);
            my_vector& output_h = this->h[h.size() - 1];

            assert(!output.empty());  // 确保输出不为空
            //printf("train-gradient\n");
            // 计算输出层的梯度
            my_vector outputLayerGradient;

            for (size_t neuronIndex = 0; neuronIndex < 1; ++neuronIndex) {
                double error = label[dataIndex] - output[neuronIndex];
                //outputLayerGradient.push_back(error * d_sigmoid(output[neuronIndex]));
                outputLayerGradient.push_back(error * d_sigmoid(output_h[neuronIndex]));
            }
            //printf("train-backward\n");
            // 反向传播
            std::vector<my_vector> layerGradients;
            layerGradients.push_back(outputLayerGradient);
            for (int layerIndex = h.size() - 2; layerIndex >= 0; --layerIndex) {
                assert(layerIndex < w.size());  // 确保权重索引在范围内
                my_vector layerGradient;
                for (size_t neuronIndex = 0; neuronIndex < h[layerIndex].size(); ++neuronIndex) {
                    double gradientSum = 0;
                    for (size_t nextLayerNeuronIndex = 0; nextLayerNeuronIndex < h[layerIndex + 1].size(); ++nextLayerNeuronIndex) {
                        assert(layerIndex < w.size() && neuronIndex < w[layerIndex].size() && nextLayerNeuronIndex < w[layerIndex][neuronIndex].size()); // 确保权重索引在范围内
                        gradientSum += w[layerIndex][neuronIndex][nextLayerNeuronIndex] * layerGradients.back()[nextLayerNeuronIndex];
                    }
                    layerGradient.push_back(gradientSum * d_sigmoid(h[layerIndex][neuronIndex]));
                }
                layerGradients.push_back(layerGradient);
            }
            //printf("train-re-new\n");
            // 更新权重和偏置
            for (size_t layerIndex = 0; layerIndex < w.size(); ++layerIndex) {
                for (size_t neuronIndex = 0; neuronIndex < w[layerIndex].size(); ++neuronIndex) {
                    for (size_t nextNeuronIndex = 0; nextNeuronIndex < w[layerIndex][neuronIndex].size(); ++nextNeuronIndex) {
                        assert(layerIndex < h.size() && neuronIndex < h[layerIndex].size());  // 确保 h 的索引在范围内
                        assert(neuronIndex < w[layerIndex].size() && nextNeuronIndex < w[layerIndex][neuronIndex].size());
                        //w[layerIndex][neuronIndex][nextNeuronIndex] += learning * h[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
                        w[layerIndex][neuronIndex][nextNeuronIndex] += learning * o[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
                        //printf("w is good!\n");
                    }
                    //b[layerIndex][neuronIndex] += learning * o[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
                    //printf("finished cal w%d%d\n", layerIndex, neuronIndex);
                }
                //printf("finish cal w%d\n", layerIndex);
                assert(layerIndex < b.size());  // 确保偏置索引在范围内
                for (size_t biasIndex = 0; biasIndex < b[layerIndex].size(); ++biasIndex) {
                    b[layerIndex][biasIndex] += learning * layerGradients[layerGradients.size() - 1 - layerIndex][biasIndex];
                    //assert(layerIndex < layerGradients.size() && layerIndex >= 0);
                    //if (biasIndex >= layerGradients[layerIndex].size())
                      //  printf("bias index(%d) is larger than the size(%d)!\n", biasIndex, layerGradients[layerIndex].size());
                    //assert(biasIndex < layerGradients[layerIndex].size());

                    //b[layerIndex][biasIndex] += learning * layerGradients[layerIndex][biasIndex];
                }
                //printf("finish cal b%d\n", layerIndex);

            }
            //printf("finish cal w\n");
        }
        // 每个epoch后输出损失
        //continue;
        if (epoch % LR_VOKE) continue;
        printf("train-printloss\n");
        double loss = 0;
        for (size_t dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
            my_vector output = forward(data[dataIndex]);
            for (size_t outputIndex = 0; outputIndex < output.size(); ++outputIndex) {
                //double error = label[dataIndex] - output[outputIndex];
                //loss += error * error;  // MSE
                loss += getMSELoss(label[dataIndex], output[outputIndex]);
            }
        }
        loss /= data.size();
        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        for (int i = 0; i < w.size(); ++i) {
            for (int j = 0; j < w[i].size(); ++j) {
                printf("w%d%d:\t", i, j);
                for (int k = 0; k < w[i][j].size(); ++k) {
                    printf("%f\t", w[i][j][k]);
                }
                printf("\nb: %f\n", b[i][j]);
            }
            printf("\n");
        }

    }

    double loss = 0;
    for (size_t dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
        my_vector output = forward(data[dataIndex]);
        for (size_t outputIndex = 0; outputIndex < output.size(); ++outputIndex) {
            //double error = label[dataIndex] - output[outputIndex];
            //loss += error * error;  // MSE
            loss += getMSELoss(label[dataIndex], output[outputIndex]);
        }
    }
    loss /= data.size();
    std::cout << "Loss: " << loss << std::endl;
}


void MyNeuron::predict(std::vector<my_vector>& test_data, my_vector& test_label)
{
    /*
    int size = test_data.size();
    double cnt = 0;
    for (double& td : test_data) {
        double pred = forward(td);
        pred = pred > 0.5 ? 1 : 0;
        cnt += isSameDouble(pred, td);
    }
    printf("correct rate is: %f\n", (double)cnt / (double)size);
    */
    //todo

}

my_vector& MyNeuron::predict(my_vector& input)
{
    // 使用forward函数获取网络的输出
    my_vector& output = forward(input);

    // 这里我们假设网络输出是二分类的概率，使用0.5作为阈值
    // 如果网络设计为多分类，可能需要选择最大值所在的索引
    // 因为我们预期输出是单个值，我们将使用output[0]作为预测概率
    //double threshold = sigmoid(0.5);

    double threshold = 0.5;
    //printf("output0 is:%f\n", output[0]);
    output[0] = (output[0] >= threshold) ? 1.0 : 0.0;

    // 返回分类决策，是h.back()的引用。
    return output;
    for (auto& wi : w) {
        for (auto& wj : wi) {
            for (auto wk : wj) {
                printf("%f\t", wk);
            }
            printf(";\t");
        }
        printf("\n");
    }

    return output; 

}

double MyNeuron::predict(my_vector& input, double threshold)
{
    //threshold = sigmoid(threshold);
    return (forward(input)[0] > threshold ? 1.0 : 0.0);
}

int MyNeuron::LR_VOKE = 500;