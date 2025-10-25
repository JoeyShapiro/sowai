#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include <ctime>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdlib>
#include <filesystem>

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

#if defined(__APPLE__)
    // Request OpenGL 4.1 Core Profile (highest on macOS)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Required on macOS
#else
    #error "glsl_version not defined for this platform"
#endif

    // Create window
    GLFWwindow* window = glfwCreateWindow(800, 600, "sowai", NULL, NULL);
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

    // Main loop
    static double last_time = 0.0;
    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
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
        unsigned char* pixels = new unsigned char[width*6 * height * 4]; // RGBA
        GLuint textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // 2. Update pixel data each frame
        // Fill your pixels array with RGBA values (0-255 each)
        for (int b = 0; b < 6; b++) {
            // Each image is 28x28, offset by b * 28 * 28
            int offset = b * 28 * 28;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    // int i = (y * width + x) * 4;
                    // Position in the combined texture (6 images side by side)
                    int tex_x = b * width + x;  // Horizontal offset for batch
                    int i = (y * width * 6 + tex_x) * 4;
                    // TODO look into all of this

                    int pixel = output_data[offset + y * width + x] * 255;
                    pixels[i + 0] = pixel; // Red
                    pixels[i + 1] = pixel; // Green
                    pixels[i + 2] = pixel; // Blue
                    pixels[i + 3] = 255; // Alpha
                }
            }
        }

        /*
        TODO
        just commit this gpu idea. it cant work, and dont like opengl
        then try just drawing pixels. not sure i am saving a lot of resources with gpu. i mean, onnx is the real problem
        then just use mfb. dont think it really goes to gpu
        */

        // Upload to GPU
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width*6, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

        // Rendering
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, display_w, display_h, 0, -1, 1); // Top-left origin
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Calculate centered position
        int textureWidth = width * 6;
        int textureHeight = height;
        float x = (display_w - textureWidth) / 2.0f;
        float y = (display_h - textureHeight) / 2.0f;

        // Draw texture
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, textureID);

        // Draw centered quad
        glBegin(GL_QUADS);
            glTexCoord2f(0, 0); glVertex2f(x, y);
            glTexCoord2f(1, 0); glVertex2f(x + textureWidth, y);
            glTexCoord2f(1, 1); glVertex2f(x + textureWidth, y + textureHeight);
            glTexCoord2f(0, 1); glVertex2f(x, y + textureHeight);
        glEnd();

        glfwSwapBuffers(window);

        // sleep until it has been 1 second since last frame
        double current_time = glfwGetTime();
        double delta_time = current_time - last_time;
        if (delta_time < 0.3) {
            // glfwWaitEventsTimeout(1.0 - delta_time);
            // ImGui_ImplGlfw_Sleep((0.3 - delta_time) * 1000.0);
        }
        last_time = current_time;
    }

    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}