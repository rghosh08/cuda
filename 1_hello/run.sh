#!/bin/bash

# Compile the CUDA program with nvcc
nvcc hello.cu -o hello

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo "Compilation successful. Executing program..."

    # Run the executable and redirect output to result.txt
    ./hello > result.txt

    echo "Output saved to result.txt."
else
    echo "Compilation failed. Exiting."
    exit 1
fi

