#include"eval.h"
#include<string>
#include<iostream>
#include<algorithm>
#include<ctime>
#include<cstdlib>

int main()
{
    srand(time(NULL));

    std::string weights_path = "/home/a1097/Project/mnist_eval/model.bin";
    std::string images_path = "/home/a1097/Project/mnist_eval/data/MNIST/raw/t10k-images-idx3-ubyte";
    std::string labels_path = "/home/a1097/Project/mnist_eval/data/MNIST/raw/t10k-labels-idx1-ubyte";

    NetParameters params;
    params = getWeighs(weights_path);

    std::vector<float> images;
    images.resize(MNIST_NINPUT * MNIST_NTEST);

    if(!mnist_image_load(images_path, images.data(), MNIST_NTEST))
    {
        std::cout << "fail to load images!" << std::endl;
    }

    const int idx = rand() % MNIST_NTEST;
    std::vector<float> test_img(images.begin() + idx * MNIST_NINPUT, images.begin() + (idx + 1) * MNIST_NINPUT);

    mnist_image_print(stdout, images.data() + idx*MNIST_NINPUT);

    Net net(params);
    std::vector<float> output = net.forward(test_img);
    int pred = std::max_element(output.begin(), output.end()) - output.begin();
    std::cout << "pred is: " << pred << std::endl;
}