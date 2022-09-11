#include <iostream>
#include <torch/torch.h>

const int64_t epochs = 1;
const int64_t batch_size = 64;
const std::string kData = R"(D:\code\C++\ml\data)";
const double_t learning_rate = 1e-3;

torch::Device get_device() {
    auto device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    return torch::Device(device_type);
}

class LeNet : public torch::nn::Module {
public:
    LeNet() : net{
            torch::nn::Sequential(
                    std::move(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 6, 5).padding(2))),
                    std::move(torch::nn::ReLU(torch::nn::ReLUOptions())),
                    std::move(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))),
                    std::move(torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, {5, 5}))),
                    std::move(torch::nn::ReLU(torch::nn::ReLUOptions())),
                    std::move(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))),
                    std::move(torch::nn::Flatten(torch::nn::FlattenOptions())),
                    std::move(torch::nn::Linear(torch::nn::LinearOptions(400, 120))),
                    std::move(torch::nn::ReLU(torch::nn::ReLUOptions())),
                    std::move(torch::nn::Dropout(torch::nn::DropoutOptions(0.5))),
                    std::move(torch::nn::Linear(torch::nn::LinearOptions(120, 84))),
                    std::move(torch::nn::ReLU(torch::nn::ReLUOptions())),
                    std::move(torch::nn::Linear(torch::nn::LinearOptions(84, 10)))
            )
    } {
        register_module("net", net);
    }

//    LeNet() : conv1(torch::nn::Conv2dOptions(1, 6, 5).padding(2)),
//              relu1(torch::nn::ReLUOptions()),
//              maxpool1(torch::nn::MaxPool2dOptions(2).stride(2)),
//              conv2(torch::nn::Conv2dOptions(6, 16, 5)),
//              relu2(torch::nn::ReLUOptions()),
//              maxpool2(torch::nn::MaxPool2dOptions(2).stride(2)),
//              flatten(torch::nn::FlattenOptions()),
//              linear1(torch::nn::LinearOptions(400, 120)),
//              relu3(torch::nn::ReLUOptions()),
//              linear2(torch::nn::LinearOptions(120, 84)),
//              relu4(torch::nn::ReLUOptions()),
//              linear3(torch::nn::LinearOptions(84, 10)) {
//        register_module("conv1", conv1);
//        register_module("relu1", relu1);
//        register_module("maxpool1", maxpool1);
//        register_module("conv2", conv2);
//        register_module("relu2", relu2);
//        register_module("maxpool2", maxpool2);
//        register_module("flatten", flatten);
//        register_module("linear", linear1);
//        register_module("relu3", relu3);
//        register_module("linear2", linear2);
//        register_module("relu4", relu4);
//        register_module("linear3", linear3);
//    }

    torch::Tensor forward(torch::Tensor x) {
        return net->forward(x);
//        auto a = conv1->forward(x);
//        auto b = relu1->forward(a);
//        auto c = maxpool1->forward(b);
//        auto d = conv2->forward(c);
//        auto e = relu2->forward(d);
//        auto f = maxpool2->forward(e);
//        auto g = flatten->forward(f);
//        auto h = linear1->forward(g);
//        auto i = relu3->forward(h);
//        auto j = linear2->forward(i);
//        auto k = relu4->forward(j);
//        auto l = linear3->forward(k);
//        auto ans = l;
//        return ans;
    }

private:
    torch::nn::Sequential net;
//    torch::nn::Conv2d conv1, conv2;
//    torch::nn::ReLU relu1, relu2, relu3, relu4;
//    torch::nn::MaxPool2d maxpool1, maxpool2;
//    torch::nn::Flatten flatten;
//    torch::nn::Linear linear1, linear2, linear3;
};

int main() {
    auto training_dataset = torch::data::datasets::MNIST(kData, torch::data::datasets::MNIST::Mode::kTrain)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());
    auto train_dataset_size = training_dataset.size().value();
    auto testing_dataset = torch::data::datasets::MNIST(kData, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());
    auto test_dataset_size = testing_dataset.size().value();

    auto training_dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(training_dataset),
            batch_size);
    auto testing_dataloader = torch::data::make_data_loader(std::move(testing_dataset), batch_size);
    auto device = get_device();

    LeNet lenet;
    lenet.to(device);

    auto loss_func = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions());
    auto optimizer = torch::optim::Adam(lenet.parameters(), torch::optim::AdamOptions(learning_rate));

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch:" << epoch + 1 << std::endl;
        //train
        {
            lenet.train();
            size_t batch_idx = 0;
            for (const auto &batch : *training_dataloader) {
                optimizer.zero_grad();

                auto data = batch.data.to(device), target = batch.target.to(device);
                auto pred = lenet.forward(data);
                auto loss = loss_func->forward(pred, target);
                loss.backward();

                optimizer.step();

                if (++batch_idx % 100 == 0) {
                    std::cout << "loss:" << loss.template item<float>() << "  "
                              << batch_idx * data.size(0) << "/"
                              << train_dataset_size << std::endl;
                }
            }
        }
        // test
        {
            torch::NoGradGuard gradGuard;
            lenet.eval();
            double_t test_loss = 0;
            int64_t correct = 0;
            for (const auto &batch : *testing_dataloader) {
                auto data = batch.data.to(device), target = batch.target.to(device);

                auto pred = lenet.forward(data);
                test_loss += loss_func->forward(pred, target).item<float>();
                correct += torch::eq(pred.argmax(1), target).sum().item<int64_t>();
            }
            test_loss /= test_dataset_size;
            std::cout << "Accuracy:" << 100 * static_cast<double>(correct) / test_dataset_size << '%' << "  "
                      << "Avg loss:" << test_loss << std::endl;
        }
    }
}