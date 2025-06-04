#!/bin/bash

# Compile CUDA and C++ programs
nvcc mat_mul_cu.cu -o mat_mul_cu
g++ mat_mul_cpp.cpp -o mat_mul_cpp

# Initialize result file
echo "CUDA Version Output:" > result.txt
./mat_mul_cu >> result.txt

echo -e "\nC++ Version Output:" >> result.txt
./mat_mul_cpp >> result.txt
