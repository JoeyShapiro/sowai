#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include <ctime>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdlib>

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
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    #error "glsl_version not defined for this platform"
#endif

    // Create window with graphics context
    float main_scale = ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor()); // Valid on GLFW 3.3+ only
    GLFWwindow* window = glfwCreateWindow((int)(1280 * main_scale), (int)(800 * main_scale), "sowai", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup scaling
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);        // Bake a fixed style scale. (until we have a solution for dynamic style scaling, changing this requires resetting Style + calling this again)
    style.FontScaleDpi = main_scale;        // Set initial font scale. (using io.ConfigDpiScaleFonts=true makes this unnecessary. We leave both here for documentation purpose)

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // Initialize random seed
    srand(time(NULL));
    
    // Create ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sowai");
    Ort::SessionOptions session_options;
    
    // Load model
    Ort::Session session(env, "generator.onnx", session_options);
    
    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // ImGui_ImplGlfw_Sleep(30);

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // put text on the frame
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | 
                        //  ImGuiWindowFlags_NoResize | 
                         ImGuiWindowFlags_NoMove | 
                         ImGuiWindowFlags_NoCollapse;

        ImGui::Begin("MyWindow", nullptr, flags);
        time_t now = time(0);
        tm* ltm = localtime(&now);
        int hour = ltm->tm_hour;
        int minute = ltm->tm_min;
        int second = ltm->tm_sec;

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
        
        // get the current time
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
        unsigned char* pixels = new unsigned char[width * height * 4]; // RGBA
        GLuint textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // 2. Update pixel data each frame
        // Fill your pixels array with RGBA values (0-255 each)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int i = (x * width + y) * 4;
                pixels[i + 0] = output_data[x * 28 + y] * 255; // Red
                pixels[i + 1] = output_data[x * 28 + y] * 255; // Green
                pixels[i + 2] = output_data[x * 28 + y] * 255; // Blue
                pixels[i + 3] = 255; // Alpha
            }
        }

        // Upload to GPU
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        ImGui::Image((void*)(intptr_t)textureID, ImVec2(width*5, height *5));

        ImGui::End();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}