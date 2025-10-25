#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include <ctime>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <chrono>
#include <thread>

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Main code
int main(int, char**)
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Create window
    GLFWwindow* window = glfwCreateWindow(800, 600, "sowai", NULL, NULL);
    if (!window) {
        printf("Failed to create window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Initialize random seed
    srand(time(NULL));
    
    // Create ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sowai");
    Ort::SessionOptions session_options;

    // Load model
    if (!std::filesystem::exists("generator.onnx")) {
        printf("model file 'generator.onnx' not found!\n");
        return 1;
    }
    
    Ort::Session session(env, "generator.onnx", session_options);
    
    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Create input tensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Input and output names
    std::vector<Ort::Value> input_tensors;
    const char* input_names[] = {"noise", "label"};
    const char* output_names[] = {"image"};

    // Calculate scale factor to fill the window width
    float scale = 800.0f / (28.0f * 6.0f);
    // Set pixel zoom to scale the image
    glPixelZoom(scale, scale);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        auto frame_start = std::chrono::high_resolution_clock::now();
        glfwPollEvents();

        time_t now = time(0);
        tm* ltm = localtime(&now);
        int hour = ltm->tm_hour%12;
        int minute = ltm->tm_min;
        int second = ltm->tm_sec;

        // Prepare noise: (6, 100)
        std::vector<float> noise(6 * 100);
        for (int i = 0; i < 6 * 100; i++) {
            noise[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        
        // Prepare labels: [0, 1, 2, 3, 4, 5]
        // TODO might be slow
        std::vector<int64_t> labels = {hour/10, hour%10, minute/10, minute%10, second/10, second%10};
        
        // Input shapes
        std::vector<int64_t> noise_shape = {6, 100};
        std::vector<int64_t> label_shape = {6};

        Ort::Value noise_tensor = Ort::Value::CreateTensor<float>(
            memory_info, noise.data(), noise.size(),
            noise_shape.data(), noise_shape.size()
        );
        
        Ort::Value label_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, labels.data(), labels.size(),
            label_shape.data(), label_shape.size()
        );
        
        // Run inference
        input_tensors.clear();
        input_tensors.push_back(std::move(noise_tensor));
        input_tensors.push_back(std::move(label_tensor));

        // auto start = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, input_tensors.data(), 2,
            output_names, 1
        );
        // std::cout << "Inference completed: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << " microseconds\n";
    
        // Get output (28x28 image)
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
    
        // 1. Create texture once (in your initialization)
        int width = 28;
        int height = 28;
        // unsigned char* pixels = new unsigned char[width*6 * height * 4]; // RGB
        unsigned char* pixels = new unsigned char[28*6 * 28 * 3];

        // 2. Update pixel data each frame
        // Fill your pixels array with RGBA values (0-255 each)
        for (int b = 0; b < 6; b++) {
            // Each image is 28x28, offset by b * 28 * 28
            int offset = b * 28 * 28;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int flipped_x = (height - 1 - x);
                    int flipped_y = (width - 1 - y);
                    // int i = (y * width + x) * 4;
                    // Position in the combined texture (6 images side by side)
                    int tex_x = b * width + flipped_x;  // Horizontal offset for batch
                    int i = (y * width * 6 + tex_x) * 3;
                    // TODO look into all of this

                    int pixel = output_data[offset + flipped_y * width + flipped_x] * 255;
                    pixels[i + 0] = pixel; // Red
                    pixels[i + 1] = pixel; // Green
                    pixels[i + 2] = pixel; // Blue
                    // pixels[i + 3] = 255; // Alpha
                }
            }
        }

        /*
        TODO
        just commit this gpu idea. it cant work, and dont like opengl
        then try just drawing pixels. not sure i am saving a lot of resources with gpu. i mean, onnx is the real problem
        then just use mfb. dont think it really goes to gpu
        */

        // Rendering
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Set up orthographic projection to match window coordinates
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, 800, 0, 600, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Calculate position to center the pixels
        float centerX = (800 - 28*6) / 2.0f;
        float centerY = (600 - 28) / 2.0f;

        // Position for drawing (raster position)
        glRasterPos2f(centerX, centerY);

        // Calculate centered position
        glDrawPixels(28*6, 28, GL_RGB, GL_UNSIGNED_BYTE, pixels);

        glfwSwapBuffers(window);

        // Calculate elapsed time and sleep if needed
        auto frame_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = frame_end - frame_start;
        
        if (elapsed.count() < 0.33) {
            std::chrono::duration<double> sleep_time(0.33 - elapsed.count());
            std::this_thread::sleep_for(sleep_time);
        }
    }

    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}