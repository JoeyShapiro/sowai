# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall
LIBS = -lglfw -framework OpenGL -framework Cocoa -framework IOKit -L/opt/homebrew/Cellar/glfw/3.4/lib

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
TARGET = imgui_app

# Include paths
INCLUDES = -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends -I/opt/homebrew/Cellar/glfw/3.4/include

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