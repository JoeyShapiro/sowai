/*
clang++ -std=c++17 -L/opt/homebrew/Cellar/onnxruntime/1.22.2_5/lib -lonnxruntime -I/opt/homebrew/Cellar/onnxruntime/1.22.2_5/include/onnxruntime onnx.cpp -o onnx_app
*/

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

int main() {
    // Initialize random seed
    srand(time(NULL));
    
    // Create ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sowai");
    Ort::SessionOptions session_options;
    
    // Load model
    Ort::Session session(env, "generator.onnx", session_options);
    
    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Prepare inputs: noise (1, 100) and label (1)
    std::vector<float> noise(100);
    for (int i = 0; i < 100; i++) {
        noise[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Random [-1, 1]
    }
    
    int64_t label = 5;  // Generate digit 5
    
    // Input shapes
    std::vector<int64_t> noise_shape = {1, 100};
    std::vector<int64_t> label_shape = {1};
    
    // Create input tensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value noise_tensor = Ort::Value::CreateTensor<float>(
        memory_info, noise.data(), noise.size(), 
        noise_shape.data(), noise_shape.size()
    );
    
    Ort::Value label_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, &label, 1,
        label_shape.data(), label_shape.size()
    );
    
    // Input and output names
    const char* input_names[] = {"noise", "label"};
    const char* output_names[] = {"image"};
    
    // Run inference
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(noise_tensor));
    input_tensors.push_back(std::move(label_tensor));
    
    std::cout << "Running inference...\n";
    // get the current time
    auto start = std::chrono::high_resolution_clock::now();
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, input_tensors.data(), 2,
        output_names, 1
    );
    std::cout << "Inference completed: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << " microseconds\n";
    
    // Get output (28x28 image)
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    
    // Print a simple ASCII representation
    std::cout << "Generated digit " << label << ":\n";
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            float pixel = output_data[i * 28 + j];
            // Map [-1, 1] to ASCII brightness
            if (pixel > 0.5) std::cout << "##";
            else if (pixel > 0.0) std::cout << "::";
            else if (pixel > -0.5) std::cout << "..";
            else std::cout << "  ";
        }
        std::cout << "\n";
    }
    
    return 0;
}
