#pragma once

#include<vector>
#include<string>
#include<math.h>
#include<memory>

#define HIDDEN_SIZE 500
#define NUM_CLASSES 10
#define INPUT_SIZE 784

#define MNIST_NINPUT 28*28
#define MNIST_NTEST 10000

struct NetParameters
{
    std::vector<float> fc1_weights;
    std::vector<float> fc1_bias;
    std::vector<float> fc2_weights;
    std::vector<float> fc2_bias;    
};

NetParameters getWeighs(const std::string file_path);
bool mnist_image_load(const std::string & fname, float * buf, const int nex);
void mnist_image_print(FILE * stream, const float * image);


std::vector<float> relu(std::vector<float>& input);
std::vector<float> fc(
    const std::vector<float>& A, // M * K
    const std::vector<float>& B, // N * K
    std::vector<float>& C, // M * N
    const std::vector<float>& bias, // M * N
    const int M, 
    const int N, 
    const int K
);


class Net
{
public:
    Net(NetParameters _paramas);
    ~Net();

    // 允许移动构造和移动赋值，禁止拷贝
    Net(Net&&) noexcept;
    Net& operator=(Net&&) noexcept;
    Net(Net&) = delete;
    Net& operator=(Net&) = delete;

    std::vector<float>forward(const std::vector<float> input);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};