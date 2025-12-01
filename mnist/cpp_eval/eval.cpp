#include"eval.h"
#include<string>
#include<iostream>
#include<vector>
#include<cstdint>
#include<fstream>
#include<thread>


// 打印图片
void mnist_image_print(FILE * stream, const float * image) {
    static_assert(MNIST_NINPUT == 28*28, "Unexpected MNIST_NINPUT");

    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            const int rgb = roundf(255.0f * image[row*28 + col]);
#ifdef _WIN32
            fprintf(stream, "%s", rgb >= 220 ? "##" : "__");                // Represented via text.
#else
            fprintf(stream, "\033[48;2;%d;%d;%dm  \033[0m", rgb, rgb, rgb); // Represented via colored blocks.
#endif // _WIN32
        }
        fprintf(stream, "\n");
    }
}

// 加载图片
bool mnist_image_load(const std::string & fname, float * buf, const int nex) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "failed to open images file %s\n", fname.c_str());
        return false;
    }
    fin.seekg(16);

    uint8_t image[MNIST_NINPUT];

    for (int iex = 0; iex < nex; ++iex) {
        fin.read((char *) image, sizeof(image));

        for (int i = 0; i < MNIST_NINPUT; ++i) {
            buf[iex*MNIST_NINPUT + i] = image[i] / 255.0f; // Normalize to [0, 1]
        }
    }

    return true;
}

// 加载权重
NetParameters getWeighs(const std::string file_path)
{   
    NetParameters params;

    const int fc1_weights_size = INPUT_SIZE * HIDDEN_SIZE;
    const int fc2_weights_size = HIDDEN_SIZE * NUM_CLASSES;

    const int fc1_bias_size = HIDDEN_SIZE;
    const int fc2_bias_size = NUM_CLASSES;

    params.fc1_weights.resize(fc1_weights_size);
    params.fc2_weights.resize(fc2_weights_size);
    params.fc1_bias.resize(fc1_bias_size);
    params.fc2_bias.resize(fc2_bias_size);
    // 以二进制模式打开文件
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open model.bin" << std::endl;
    }

    // 读取 fc1.weight 的数据
    // file.read() 需要一个 char* 指针和要读取的字节数
    // sizeof(float) 是 4, fc1_size 是浮点数的数量
    file.read(reinterpret_cast<char*>(params.fc1_weights.data()), fc1_weights_size * sizeof(float));
    file.read(reinterpret_cast<char*>(params.fc1_bias.data()), fc1_bias_size * sizeof(float));
    file.read(reinterpret_cast<char*>(params.fc2_weights.data()), fc2_weights_size * sizeof(float));
    file.read(reinterpret_cast<char*>(params.fc2_bias.data()), fc2_bias_size * sizeof(float));

    if (!file) {
        std::cerr << "Error reading fc1 weights from file." << std::endl;
        // gcount() 会返回实际读取的字节数，用于调试
        std::cerr << "Read " << file.gcount() << " bytes." << std::endl;
    }
    
    file.close();
    return params;
}




struct Net::Impl
{
    NetParameters params;
    Impl(NetParameters _params) : params(_params) {};
};


Net::Net(NetParameters _params) : pImpl(std::make_unique<Impl>(_params)) {};
Net::~Net() = default;
Net::Net(Net&&) noexcept = default;
Net& Net::operator=(Net&&) noexcept = default;


std::vector<float> fc(
    // M 是批量大小
    const std::vector<float>& A, // M * K
    const std::vector<float>& B, // N * K
    std::vector<float>& C, // M * N
    const std::vector<float>& bias, // M * N
    const int M, 
    const int N, 
    const int K
)
{   
    // 一次只推理一张图片
    if (M != 1) return C;

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    const int size = (N + num_threads - 1) / num_threads;

    for (int i = 0; i < num_threads; ++i)
    {
        int start = i * size;
        int end = (i + 1) * size;
        if (end > N) end = N;

        threads.emplace_back([&, start, end]
        {
            for (int n = start; n < end; ++n)
            {
                float sum = bias[n];
                for (int k = 0; k < K; ++k)
                {
                    sum += A[k] * B[n * K + k];
                }
                C[n] = sum;
            }
        });
    }

    for (auto& t : threads)
    {
        t.join();
    }

    return C;
}

std::vector<float> relu(std::vector<float>& input)
{
    for (int i = 0; i < input.size(); ++i)
    {
        input[i] = fmax(input[i], 0);
    }
    return input;
}


std::vector<float> Net::forward(std::vector<float> input)
{
    std::vector<float> fc1_input = input; 
    std::vector<float> fc1_output(HIDDEN_SIZE, 0);
    
    fc1_output = fc(fc1_input, pImpl->params.fc1_weights, fc1_output, pImpl->params.fc1_bias, 1, HIDDEN_SIZE, INPUT_SIZE);

    fc1_output = relu(fc1_output); 
    
    std::vector<float> fc2_input = fc1_output;
    std::vector<float> final_output(NUM_CLASSES, 0);

    final_output = fc(fc2_input, pImpl->params.fc2_weights, final_output, pImpl->params.fc2_bias, 1, NUM_CLASSES, HIDDEN_SIZE);

    return final_output;
};