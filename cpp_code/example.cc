#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cuda_runtime.h>
#include "utils.h"
/* 超参数，与转换后的onnx模型保持一致 */
int INPUT_H=224;
int INPUT_W=224;
int OUTPUT_SIZE=1000;
int batchSize = 4;

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) override {
        using namespace std;
        string s;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                s = "INTERNAL_ERROR";
                break;
            case Severity::kERROR:
                s = "ERROR";
                break;
            case Severity::kWARNING:
                s = "WARNING";
                break;
            case Severity::kINFO:
                s = "INFO";
                break;
            case Severity::kVERBOSE:
                s = "VERBOSE";
                break;
        }
        cerr << s << ": " << msg << endl;
    }
};

/* 自销毁定义，使用方便 */
template<typename T>
struct Destroy {
    void operator()(T *t) const {
        t->destroy();
    }
};

/// Optional : Print dimensions as string
std::string printDim(const nvinfer1::Dims & d) {
    using namespace std;
    ostringstream oss;
    for (int j = 0; j < d.nbDims; ++j) {
        oss << d.d[j];
        if (j < d.nbDims - 1)
            oss << "x";
    }
    return oss.str();
}

/* 用于从onxx中读取模型，解析模型--本example未使用 */
nvinfer1::ICudaEngine *createCudaEngine(const std::string &onnxFileName, nvinfer1::ILogger &logger, int batchSize) {
    using namespace std;
    using namespace nvinfer1;

    /*IBuilder对象记录log*/
    unique_ptr<IBuilder, Destroy<IBuilder>> builder{createInferBuilder(logger)};
    /*定义网络类型，参数表示固定batch以及INPUT动态输入 */
    unique_ptr<INetworkDefinition, Destroy<INetworkDefinition>> network{
            builder->createNetworkV2(1U << (unsigned) NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};
    /*onnxParse解析器，将onnx模型转化为engine*/
    unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{
            nvonnxparser::createParser(*network, logger)};

    if (!parser->parseFromFile(onnxFileName.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
        throw runtime_error("ERROR: could not parse ONNX model " + onnxFileName + " !");

    // Create Optimization profile and set the batch size
    // This profile will be valid for all images whose size falls in the range
    // but TensorRT will optimize for {batchSize, 3, 422,422}
    // We do not need to check the return of setDimension and addOptimizationProfile here as all dims are explicitly set
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions("input", OptProfileSelector::kMIN, Dims4{batchSize, 3, 1, 1});
    profile->setDimensions("input", OptProfileSelector::kMAX, Dims4{batchSize, 3, 422,422});
    profile->setDimensions("input", OptProfileSelector::kOPT, Dims4{batchSize, 3, INPUT_H, INPUT_W});
    
    // Create a builder configuration object.
    unique_ptr<IBuilderConfig, Destroy<IBuilderConfig>> config(builder->createBuilderConfig());
    config->addOptimizationProfile(profile);
    return builder->buildEngineWithConfig(*network, *config);
}

/* 前向推理，将输入复制到gpu中，前向，取输出 */
void launchInference(nvinfer1::IExecutionContext *context, cudaStream_t stream, float * input,
                     float * output, void **buffers, int batchSize) {
    
    int inputId = 0, outputId = 1;

    cudaMemcpyAsync(buffers[inputId], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice,
                    stream);

    context->enqueueV2(buffers, stream, nullptr);

    cudaMemcpyAsync(output, buffers[outputId], batchSize * OUTPUT_SIZE * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
}


int PostProcess(std::vector<float>& cpu_output) {
    auto classes = getClassNames("../data/imagenet_classes.txt");

        // calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return std::exp(val);});
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);
    // find top classes predicted by the model
    std::vector< int > indices(cpu_output.size());
    // generate sequence 0, 1, 2, 3, ..., 999
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {return cpu_output[i1] > cpu_output[i2];});
    // print results
    int i = 0;
    while (cpu_output[indices[i]] / sum > 0.005)
    {
        if (classes.size() > indices[i])
        {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "n";
        ++i;
    }
    return 0;
}

int main(int argc, char **argv)
{
    using namespace std;
    using namespace nvinfer1;

    Logger logger;
    Logger logger_t;
    
    logger.log(ILogger::Severity::kINFO, "C++ TensorRT (almost) minimal example !!! ");
    logger.log(ILogger::Severity::kINFO, "Creating engine ...");
    // onnx解析成engine
    // In order to create an object of type IExecutionContext, first create an object of type ICudaEngine (the engine).
    int batchSize = 4;
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine(createCudaEngine("../data/resnet50_pytorch_bs4.onnx", logger_t, batchSize)); 

    /* 从二进制文件trt中取模型流并生成engine 
    ifstream file("resnet_engine.trt", ios::binary);

    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    std::cout<<"size "<<size<<endl;

    file.seekg(0, file.beg);
    trtModelStream = new char[size];

    logger.log(ILogger::Severity::kINFO, "2");
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    unique_ptr<IRuntime, Destroy<IRuntime>> runtime(createInferRuntime(logger));
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine(runtime->deserializeCudaEngine(trtModelStream, size, nullptr));
//*/

    /* 可查看该engine输入输出维度 */
    // Optional : Print all bindings : name + dims + dtype
    cout << "=============\nBindings :\n";
    int n = engine->getNbBindings();
    for (int i = 0; i < n; ++i) {
        Dims d = engine->getBindingDimensions(i);
        cout << i << " : " << engine->getBindingName(i) << " : dims=" << printDim(d);
        cout << " , dtype=" << (int) engine->getBindingDataType(i) << " ";
        cout << (engine->bindingIsInput(i) ? "IN" : "OUT") << endl;
    }
    cout << "=============\n\n";

    /* In order to run inference, use the interface IExecutionContext */
    logger.log(ILogger::Severity::kINFO, "Creating context ...");
    unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context(engine->createExecutionContext());
    context->setBindingDimensions(0, Dims4(batchSize, 3, INPUT_H, INPUT_W));

    /* Create an asynchronous stream */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* 输入输出 buffer */
    float data[batchSize * 3 * INPUT_W * INPUT_H];
    float prob[batchSize * OUTPUT_SIZE];
    
    /* 输入输出 cuda buffer */
    void* buffers[2]{0};
    cudaMalloc(&buffers[0], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc(&buffers[1], batchSize * OUTPUT_SIZE * sizeof(float));

    logger.log(ILogger::Severity::kINFO, "---------------------prepare ok-------------------------");

    /* 输入前处理 */
    std::string img_dir = std::string(argv[1]);
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    logger.log(ILogger::Severity::kINFO, "---------------------read files ok-------------------------");

    int fcount = 0;
    for (int f = 0; f < (int)file_names.size(); f++) {
        fcount++;
        if (fcount < batchSize && f + 1 != (int)file_names.size()) continue;
        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) continue;
            cv::Mat pr_img = preprocess_img2(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                // uchar* uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3f>(row,col)[0];
                    data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] =  pr_img.at<cv::Vec3f>(row,col)[1] ;
                    data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] =  pr_img.at<cv::Vec3f>(row,col)[2] ;
                    ++i;
                }
            }
        }
    }

    logger.log(ILogger::Severity::kINFO, "---------------------read imgs ok-------------------------");

    /* 前向推理 */
    cout << "Running the inference !" << endl;
    launchInference(context.get(), stream, data, prob, buffers, batchSize);
    cudaStreamSynchronize(stream);

    /* 输出 */
    /* 高维特征 */
    std::vector<float> output_single_img(prob, prob+ OUTPUT_SIZE);
    PostProcess(output_single_img);

    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    return 0;

}