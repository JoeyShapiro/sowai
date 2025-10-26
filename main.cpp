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

    int window_width = 800;
    int window_height = 600;
    auto gen_timer = std::chrono::milliseconds(333);
    auto frame_timer = std::chrono::milliseconds(120);
    auto busy_timer = std::chrono::milliseconds(120);

    // Create window
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "sowai", NULL, NULL);
    if (!window) {
        printf("Failed to create window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

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

    int img_width = 28;
    int img_height = 28;
    int batch_size = 6; // TODO could try to batch multiple time samples all at once for performance

    unsigned char* pixels = new unsigned char[img_width * batch_size * img_height * 3];
    float* noise = new float[batch_size * 100];
    // Prepare labels: [0, 1, 2, 3, 4, 5]
    int64_t* labels = new int64_t[batch_size];

    // Input shapes
    int64_t* noise_shape = new int64_t[2]{batch_size, 100};
    int64_t* label_shape = new int64_t[1]{batch_size};

    // Main loop
    auto last_gen = std::chrono::high_resolution_clock::now();
    auto last_frame = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window))
    {
        auto frame_start = std::chrono::high_resolution_clock::now();
        glfwPollEvents();

        if (frame_start - last_gen >= gen_timer) {
            last_gen = frame_start;

            time_t now = time(0);
            tm* ltm = localtime(&now);
            int hour = ltm->tm_hour%12;
            int minute = ltm->tm_min;
            int second = ltm->tm_sec;
            labels[0] = hour / 10;
            labels[1] = hour % 10;
            labels[2] = minute / 10;
            labels[3] = minute % 10;
            labels[4] = second / 10;
            labels[5] = second % 10;
    
            // Prepare noise: (6, 100)
            for (int i = 0; i < 6 * 100; i++) {
                noise[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            }
    
            Ort::Value noise_tensor = Ort::Value::CreateTensor<float>(
                memory_info, noise, batch_size * 100,
                noise_shape, 2
            );
            
            Ort::Value label_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, labels, batch_size,
                label_shape, 1
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
            // printf("Inference completed: %lld milliseconds\n",
            //        std::chrono::duration_cast<std::chrono::milliseconds>(
            //            std::chrono::high_resolution_clock::now() - start).count()
            // );
        
            // Get output (28x28 image)
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
    
            // 2. Update pixel data each frame
            // Fill your pixels array with RGBA values (0-255 each)
            for (int b = 0; b < batch_size; b++) {
                // Each image is 28x28, offset by b * 28 * 28
                int offset = b * img_height * img_width;
                
                for (int y = 0; y < img_height; y++) {
                    for (int x = 0; x < img_width; x++) {
                        int flipped_x = (img_height - 1 - x);
                        int flipped_y = (img_width - 1 - y);
    
                        // Position in the combined texture (6 images side by side)
                        int tex_x = b * img_width + flipped_x;  // Horizontal offset for batch
                        int i = (y * img_width * batch_size + tex_x) * 3;
    
                        int pixel = output_data[offset + flipped_y * img_width + flipped_x] * 255;
                        pixels[i + 0] = pixel; // Red
                        pixels[i + 1] = pixel; // Green
                        pixels[i + 2] = pixel; // Blue
                    }
                }
            }
        }

        if (frame_start - last_frame >= frame_timer) {
            last_frame = frame_start;
            glfwGetFramebufferSize(window, &window_width, &window_height);
            // Calculate scale factor to fill the window width
            float scale = (float)window_width / (img_width * batch_size);
            // Set pixel zoom to scale the image
            glPixelZoom(scale, scale);
    
            // Calculate position to center the pixels
            float centerX = (window_width - img_width * batch_size * scale) / 2.0f;
            float centerY = (window_height - img_height * scale) / 2.0f;
    
            // Position for drawing (raster position)
            glRasterPos2f(centerX, centerY);
    
            // Rendering
            // glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
            glClear(GL_COLOR_BUFFER_BIT);
    
            // Set up orthographic projection to match window coordinates
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0, window_width, 0, window_height, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
    
            // Calculate centered position
            glDrawPixels(img_width * batch_size, img_height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    
            glfwSwapBuffers(window);
        } else {
            std::this_thread::sleep_for(busy_timer);
        }
    }

    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}