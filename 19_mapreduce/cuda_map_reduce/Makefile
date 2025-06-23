# CUDA MapReduce Makefile
# Compiler and flags
NVCC = nvcc
CFLAGS = -O3 -std=c++11
LIBS = -lcublas -lcurand

# Project files
TARGET = map_reduce_cu
HEADER = mapreduce_cu.h
SOURCE = mapreduce_cu.cu
OBJECTS = $(SOURCE:.cu=.o)

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(SOURCE) $(HEADER)
	$(NVCC) $(CFLAGS) $(SOURCE) -o $(TARGET) $(LIBS)

# Build object files (if needed for separate compilation)
%.o: %.cu $(HEADER)
	$(NVCC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(TARGET) $(OBJECTS) *.o

# Install (copy to system path - optional)
install: $(TARGET)
	sudo cp $(TARGET) /usr/local/bin/

# Uninstall
uninstall:
	sudo rm -f /usr/local/bin/$(TARGET)

# Run with default parameters
run: $(TARGET)
	./$(TARGET)

# Run with different dataset sizes
test-small: $(TARGET)
	./$(TARGET) 1000000

test-medium: $(TARGET)
	./$(TARGET) 100000000

test-large: $(TARGET)
	./$(TARGET) 1000000000

# Debug build with more verbose output
debug: CFLAGS += -g -G
debug: $(TARGET)

# Help target
help:
	@echo "CUDA MapReduce Makefile"
	@echo "======================="
	@echo "Targets:"
	@echo "  all          - Build the executable (default)"
	@echo "  clean        - Remove build files"
	@echo "  run          - Build and run with default settings"
	@echo "  test-small   - Build and run with 1M elements"
	@echo "  test-medium  - Build and run with 100M elements"
	@echo "  test-large   - Build and run with 1B elements"
	@echo "  debug        - Build with debug symbols"
	@echo "  install      - Install to system path"
	@echo "  uninstall    - Remove from system path"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make                    # Build"
	@echo "  make run                # Build and run"
	@echo "  make test-large         # Test with 1B elements"
	@echo "  ./$(TARGET) 50000000    # Run with custom size"

# Declare phony targets
.PHONY: all clean run test-small test-medium test-large debug install uninstall help
