# Compiler and flags
CXX = clang++
CXXFLAGS = -std=c++23 -Wall -ggdb
LIBS = -lglfw -framework OpenGL -framework Cocoa -framework IOKit -L/opt/homebrew/Cellar/glfw/3.4/lib \
	   -L/opt/homebrew/Cellar/onnxruntime/1.22.2_5/lib -lonnxruntime

# Directories
IMGUI_DIR = imgui
SRC_DIR = .

# ImGui source files
IMGUI_SOURCES = $(IMGUI_DIR)/imgui.cpp \
                $(IMGUI_DIR)/imgui_demo.cpp \
                $(IMGUI_DIR)/imgui_draw.cpp \
                $(IMGUI_DIR)/imgui_tables.cpp \
                $(IMGUI_DIR)/imgui_widgets.cpp \
                $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp \
                $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp

# Your application source files
APP_SOURCES = main.cpp

# All sources
SOURCES = $(IMGUI_SOURCES) $(APP_SOURCES)

# Object files
OBJS = $(SOURCES:.cpp=.o)

# Output executable
TARGET = sowai

# Include paths
INCLUDES = -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends -I/opt/homebrew/Cellar/glfw/3.4/include \
		   -I/opt/homebrew/Cellar/onnxruntime/1.22.2_5/include/onnxruntime

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run