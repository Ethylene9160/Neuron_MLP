cmake_minimum_required(VERSION 3.0)
project(MyNeuronProject)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 创建动态链接库（DLL）
add_library(MyNeuron SHARED MyNeuron.cpp)

# 如果您的项目依赖于其他头文件目录，使用include_directories添加
# include_directories(MyNeuron PRIVATE MyNeuron.h)

# 设置生成的DLL的输出目录
set_target_properties(MyNeuron PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
