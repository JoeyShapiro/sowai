# Sowai
This is a clock app that generates the digits using Gen AI.
This implementation uses MNIST and a custom CGAN format. It is good enough
This is based on [sowon](https://github.com/tsoding/sowon)

Video demoing it: https://www.youtube.com/watch?v=53GhsWzmWLw

## Requirements
- GLFW
- ONNX Runtime
- a model named 'generator.onnx'

## Possible Improvements
- use something like MiniFB to remove the dependency on OpenGL and GLFW
- batch multiple seconds of gen to improve performance
- allow for configuration flags
- make gen timer dynamic based on how much it can do it 1 second
- get a better dataset for higher quality numbers
- add the colon symbols to the timer
- add a timer mode

## Demo
[demo](demo.gif)
