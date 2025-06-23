#!/bin/bash

# ================================================================
# CUDA MapReduce - Complete Management Script
# ================================================================
# Usage: ./mapreduce.sh [command] [options]
# Author: Generated for CUDA MapReduce Project

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Program configuration
PROGRAM="./map_reduce_cu"
HEADER_FILE="mapreduce_cu.h"
SOURCE_FILE="mapreduce_cu.cu"
MAKEFILE="Makefile"

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}           CUDA MapReduce - Management Script${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
}

print_separator() {
    echo -e "${BLUE}----------------------------------------------------------------${NC}"
}

log_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_step() {
    echo -e "${CYAN}🔄 $1${NC}"
}

# ================================================================
# HELP AND USAGE FUNCTIONS
# ================================================================

print_usage() {
    echo -e "${CYAN}${BOLD}USAGE:${NC}"
    echo -e "  $0 [command] [options]"
    echo ""
    echo -e "${CYAN}${BOLD}COMMANDS:${NC}"
    echo -e "  ${GREEN}help${NC}              Show this help message"
    echo -e "  ${GREEN}setup${NC}             Initial project setup"
    echo -e "  ${GREEN}build${NC}             Build the MapReduce program"
    echo -e "  ${GREEN}run${NC} [size]        Run with specified dataset size"
    echo -e "  ${GREEN}test${NC}              Run predefined test suite"
    echo -e "  ${GREEN}benchmark${NC}         Run comprehensive benchmarks"
    echo -e "  ${GREEN}check${NC}             Check system requirements"
    echo -e "  ${GREEN}examples${NC}          Show usage examples"
    echo -e "  ${GREEN}clean${NC}             Clean build files"
    echo -e "  ${GREEN}monitor${NC}           Monitor GPU during execution"
    echo ""
    echo -e "${CYAN}${BOLD}DIRECT PROGRAM USAGE:${NC}"
    echo -e "  ${PROGRAM} [dataset_size]"
    echo -e "  dataset_size: 1,000 to 100,000,000,000 (100B) elements"
    echo ""
}

print_examples() {
    echo -e "${CYAN}${BOLD}EXAMPLES:${NC}"
    echo ""
    echo -e "${YELLOW}${BOLD}Basic Usage:${NC}"
    echo -e "  $0 run                    # Default 10M elements"
    echo -e "  $0 run 1000000           # 1M elements (~4MB memory)"
    echo -e "  $0 run 100000000         # 100M elements (~400MB memory)"
    echo -e "  $0 run 1000000000        # 1B elements (~4GB memory)"
    echo -e "  $0 run 5000000000        # 5B elements (~20GB memory)"
    echo ""
    echo -e "${YELLOW}${BOLD}Performance Testing:${NC}"
    echo -e "  $0 test                  # Run predefined test suite"
    echo -e "  $0 benchmark             # Comprehensive benchmarks"
    echo -e "  $0 monitor               # Monitor GPU while running"
    echo ""
    echo -e "${YELLOW}${BOLD}Project Management:${NC}"
    echo -e "  $0 setup                 # Initial setup and build"
    echo -e "  $0 build                 # Build from source"
    echo -e "  $0 check                 # Check GPU and system"
    echo -e "  $0 clean                 # Clean build files"
    echo ""
    echo -e "${YELLOW}${BOLD}Direct Program Usage:${NC}"
    echo -e "  ${PROGRAM}                        # Default settings"
    echo -e "  ${PROGRAM} 50000000              # 50M elements"
    echo -e "  ${PROGRAM} --help                # Show program help"
    echo ""
    echo -e "${YELLOW}${BOLD}Memory Guidelines for A10G (24GB GPU):${NC}"
    echo -e "  • ${GREEN}Safe${NC}: Up to 5B elements (~20GB memory)"
    echo -e "  • ${YELLOW}Caution${NC}: 5-6B elements (near GPU memory limit)"
    echo -e "  • ${RED}Risk${NC}: >6B elements (likely to fail)"
    echo ""
    echo -e "${CYAN}${BOLD}Performance Expectations:${NC}"
    echo -e "  • Small (1M-10M): Sub-second execution"
    echo -e "  • Medium (100M-1B): Seconds to minutes"
    echo -e "  • Large (1B-5B): Minutes (depending on algorithm)"
    echo ""
}

# ================================================================
# SYSTEM CHECK FUNCTIONS
# ================================================================

check_system() {
    echo -e "${CYAN}${BOLD}SYSTEM REQUIREMENTS CHECK:${NC}"
    echo ""
    
    local all_good=true
    
    # Check if source files exist
    print_separator
    echo -e "${CYAN}Source Files:${NC}"
    if [ -f "$HEADER_FILE" ]; then
        log_info "Header file found: $HEADER_FILE"
    else
        log_error "Header file missing: $HEADER_FILE"
        all_good=false
    fi
    
    if [ -f "$SOURCE_FILE" ]; then
        log_info "Source file found: $SOURCE_FILE"
    else
        log_error "Source file missing: $SOURCE_FILE"
        all_good=false
    fi
    
    # Check if CUDA program exists
    if [ -f "$PROGRAM" ]; then
        log_info "MapReduce program compiled: $PROGRAM"
    else
        log_warning "MapReduce program not compiled yet"
        echo -e "   Run: $0 build"
    fi
    echo ""
    
    # Check NVIDIA GPU and drivers
    print_separator
    echo -e "${CYAN}GPU and CUDA:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA drivers installed"
        echo ""
        echo -e "${CYAN}GPU Information:${NC}"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,utilization.memory --format=csv,noheader,nounits | while IFS=',' read -r idx name total free used temp gpu_util mem_util; do
            echo -e "  GPU $idx: ${GREEN}$name${NC}"
            echo -e "    Memory: ${GREEN}${free}MB free${NC} / ${total}MB total (${used}MB used)"
            echo -e "    Usage: ${GREEN}${gpu_util}% GPU${NC}, ${GREEN}${mem_util}% Memory${NC}, ${GREEN}${temp}°C${NC}"
        done
        echo ""
    else
        log_error "nvidia-smi not found"
        echo -e "   CUDA drivers may not be installed"
        all_good=false
        echo ""
    fi
    
    # Check CUDA compiler
    if command -v nvcc &> /dev/null; then
        nvcc_version=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
        log_info "NVCC (CUDA compiler) available - Version: $nvcc_version"
    else
        log_error "NVCC not found"
        echo -e "   CUDA toolkit may not be installed"
        all_good=false
    fi
    echo ""
    
    # Check system memory
    print_separator
    echo -e "${CYAN}System Memory:${NC}"
    free -h | head -n 2 | while read line; do
        echo -e "  ${GREEN}$line${NC}"
    done
    echo ""
    
    # Check disk space
    echo -e "${CYAN}Disk Space (current directory):${NC}"
    df -h . | tail -n 1 | while read line; do
        echo -e "  ${GREEN}$line${NC}"
    done
    echo ""
    
    # Final status
    print_separator
    if [ "$all_good" = true ]; then
        log_info "System check passed! Ready to run CUDA MapReduce."
    else
        log_error "System check found issues. Please resolve them before proceeding."
    fi
    echo ""
}

check_memory_for_size() {
    local n=$1
    local memory_gb=$(echo "scale=2; $n * 4 / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "0")
    
    if command -v nvidia-smi &> /dev/null; then
        local free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
        local free_gb=$(echo "scale=2; $free_mem / 1024" | bc -l 2>/dev/null || echo "0")
        
        echo -e "${CYAN}Memory Analysis:${NC}"
        echo -e "  Required: ${YELLOW}${memory_gb} GB${NC}"
        echo -e "  Available: ${GREEN}${free_gb} GB${NC}"
        
        if (( $(echo "$memory_gb > $free_gb * 0.8" | bc -l 2>/dev/null || echo 0) )); then
            log_warning "Dataset requires ${memory_gb}GB but only ${free_gb}GB available!"
            echo -e "   Consider using a smaller dataset"
            return 1
        elif (( $(echo "$memory_gb > $free_gb * 0.5" | bc -l 2>/dev/null || echo 0) )); then
            log_warning "Large dataset will use $(echo "scale=1; $memory_gb / $free_gb * 100" | bc -l 2>/dev/null || echo "?")% of GPU memory"
        fi
    fi
    echo ""
    return 0
}

# ================================================================
# BUILD AND SETUP FUNCTIONS
# ================================================================

setup_project() {
    echo -e "${CYAN}${BOLD}PROJECT SETUP:${NC}"
    echo ""
    
    # Make this script executable if not already
    if [ ! -x "$0" ]; then
        chmod +x "$0"
        log_info "Made script executable"
    fi
    
    # Check source files
    if [ ! -f "$HEADER_FILE" ] || [ ! -f "$SOURCE_FILE" ]; then
        log_error "Source files not found!"
        echo -e "   Required files: $HEADER_FILE, $SOURCE_FILE"
        echo -e "   Make sure they are in the current directory"
        return 1
    fi
    
    log_info "Source files found"
    
    # Build the program
    build_program
    
    if [ $? -eq 0 ]; then
        echo ""
        log_info "Setup complete! Ready to use CUDA MapReduce."
        echo ""
        echo -e "${CYAN}Quick start:${NC}"
        echo -e "  $0 run           # Run with default settings"
        echo -e "  $0 test          # Run test suite"
        echo -e "  $0 examples      # Show detailed examples"
        echo ""
    else
        log_error "Setup failed during build phase"
        return 1
    fi
}

build_program() {
    echo -e "${CYAN}${BOLD}BUILDING MAPREDUCE PROGRAM:${NC}"
    echo ""
    
    if [ -f "$MAKEFILE" ]; then
        log_step "Using Makefile build system..."
        make clean && make
    else
        log_step "Using direct nvcc compilation..."
        nvcc -O3 -std=c++11 "$SOURCE_FILE" -o map_reduce_cu -lcublas -lcurand
    fi
    
    if [ $? -eq 0 ]; then
        log_info "Build successful!"
        return 0
    else
        log_error "Build failed!"
        echo -e "   Check that CUDA toolkit is properly installed"
        echo -e "   Run: $0 check"
        return 1
    fi
}

clean_files() {
    echo -e "${CYAN}${BOLD}CLEANING BUILD FILES:${NC}"
    echo ""
    
    if [ -f "$MAKEFILE" ]; then
        make clean
    else
        rm -f map_reduce_cu *.o
        log_info "Removed object files and executable"
    fi
    
    # Clean benchmark results
    if [ -f "benchmark_results.csv" ]; then
        rm -f benchmark_results.csv
        log_info "Removed benchmark results"
    fi
    
    log_info "Cleanup completed"
    echo ""
}

# ================================================================
# EXECUTION FUNCTIONS
# ================================================================

run_with_size() {
    local size=$1
    
    if [ ! -f "$PROGRAM" ]; then
        log_error "Program not found. Building first..."
        build_program || return 1
    fi
    
    if [ -z "$size" ]; then
        echo -e "${YELLOW}No size specified, using default (10M elements)${NC}"
        size=""
    else
        # Validate size if provided
        if ! [[ "$size" =~ ^[0-9]+$ ]] || [ "$size" -lt 1000 ] || [ "$size" -gt 100000000000 ]; then
            log_error "Invalid size: $size"
            echo -e "   Size must be between 1,000 and 100,000,000,000"
            return 1
        fi
        
        # Check memory requirements
        check_memory_for_size "$size"
    fi
    
    echo -e "${CYAN}Running MapReduce with ${GREEN}${size:-default}${NC} ${CYAN}elements...${NC}"
    echo ""
    
    $PROGRAM $size
}

run_tests() {
    echo -e "${CYAN}${BOLD}RUNNING TEST SUITE:${NC}"
    echo ""
    
    if [ ! -f "$PROGRAM" ]; then
        log_error "Program not found. Building first..."
        build_program || return 1
    fi
    
    # Test sizes (elements:label:description)
    local test_sizes=(
        "1000000:1M:Small test"
        "10000000:10M:Default size" 
        "100000000:100M:Medium test"
        "1000000000:1B:Large test"
    )
    
    for test in "${test_sizes[@]}"; do
        IFS=':' read -r size label description <<< "$test"
        print_separator
        echo -e "${YELLOW}${BOLD}Running $description ($label elements)...${NC}"
        echo ""
        
        # Check if we have enough memory
        if ! check_memory_for_size "$size"; then
            log_warning "Skipping $description due to memory constraints"
            continue
        fi
        
        $PROGRAM $size
        echo ""
        log_info "$description completed"
        echo ""
    done
    
    log_info "Test suite completed"
}

run_benchmark() {
    echo -e "${CYAN}${BOLD}COMPREHENSIVE BENCHMARK SUITE:${NC}"
    echo ""
    
    if [ ! -f "$PROGRAM" ]; then
        log_error "Program not found. Building first..."
        build_program || return 1
    fi
    
    # Benchmark sizes
    local bench_sizes=(
        "1000000:1M"
        "5000000:5M"
        "10000000:10M"
        "50000000:50M"
        "100000000:100M"
        "500000000:500M"
        "1000000000:1B"
        "2000000000:2B"
        "5000000000:5B"
    )
    
    local results_file="benchmark_results_$(date +%Y%m%d_%H%M%S).csv"
    
    log_step "Running benchmarks, results will be saved to: $results_file"
    echo ""
    echo "Size,Elements,Custom_Time_ms,Thrust_Time_ms,Speedup,Throughput_M_per_sec" > "$results_file"
    
    for bench in "${bench_sizes[@]}"; do
        IFS=':' read -r size label <<< "$bench"
        
        # Check memory before running
        if ! check_memory_for_size "$size" > /dev/null 2>&1; then
            log_warning "Skipping $label elements due to memory constraints"
            continue
        fi
        
        echo -e "${YELLOW}Benchmarking $label elements...${NC}"
        
        # Run and capture output, parse timing results
        local output=$($PROGRAM $size 2>/dev/null)
        local custom_time=$(echo "$output" | grep "Custom MapReduce time:" | awk '{print $4}')
        local thrust_time=$(echo "$output" | grep "Thrust MapReduce time:" | awk '{print $4}')
        local speedup=$(echo "$output" | grep "Speedup:" | awk '{print $2}' | tr -d 'x')
        local throughput=$(echo "$output" | grep "Throughput:" | awk '{print $2}')
        
        # Save to CSV
        echo "$label,$size,$custom_time,$thrust_time,$speedup,$throughput" >> "$results_file"
        
        echo -e "  Custom: ${custom_time}ms, Thrust: ${thrust_time}ms, Speedup: ${speedup}x"
        log_info "$label completed"
        echo ""
    done
    
    echo ""
    log_info "Benchmark results saved to $results_file"
    
    if command -v python3 &> /dev/null && python3 -c "import matplotlib" 2>/dev/null; then
        echo ""
        echo -e "${CYAN}Python with matplotlib detected. Generate plots? (y/n):${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            generate_plots "$results_file"
        fi
    fi
}

monitor_gpu() {
    echo -e "${CYAN}${BOLD}GPU MONITORING MODE:${NC}"
    echo -e "Press Ctrl+C to stop monitoring"
    echo ""
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not available"
        return 1
    fi
    
    # Start monitoring in background if a program is running
    if pgrep -f "$PROGRAM" > /dev/null; then
        log_info "MapReduce program detected, monitoring active processes..."
    else
        log_info "No active MapReduce program, monitoring GPU status..."
    fi
    
    echo ""
    while true; do
        clear
        echo -e "${CYAN}${BOLD}GPU Status - $(date)${NC}"
        print_separator
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits | while IFS=',' read -r idx name temp gpu_util mem_util mem_used mem_total power; do
            echo -e "GPU $idx: ${GREEN}$name${NC}"
            echo -e "  Temperature: ${GREEN}${temp}°C${NC}"
            echo -e "  Utilization: ${GREEN}${gpu_util}% GPU${NC}, ${GREEN}${mem_util}% Memory${NC}"
            echo -e "  Memory: ${GREEN}${mem_used}MB / ${mem_total}MB${NC}"
            echo -e "  Power: ${GREEN}${power}W${NC}"
            echo ""
        done
        
        # Check for active processes
        local processes=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits 2>/dev/null)
        if [ -n "$processes" ]; then
            echo -e "${CYAN}Active GPU Processes:${NC}"
            echo "$processes" | while IFS=',' read -r pid name memory; do
                echo -e "  PID $pid: ${GREEN}$name${NC} (${memory}MB)"
            done
        else
            echo -e "${YELLOW}No active GPU processes${NC}"
        fi
        
        sleep 2
    done
}

generate_plots() {
    local csv_file=$1
    local plot_script="plot_results.py"
    
    cat > "$plot_script" << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print("Usage: python3 plot_results.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Execution Time Comparison
ax1.plot(df['Elements'], df['Custom_Time_ms'], 'bo-', label='Custom Implementation')
ax1.plot(df['Elements'], df['Thrust_Time_ms'], 'ro-', label='Thrust Implementation')
ax1.set_xlabel('Dataset Size (Elements)')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Execution Time Comparison')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True)

# Plot 2: Speedup
ax2.plot(df['Elements'], df['Speedup'], 'go-')
ax2.set_xlabel('Dataset Size (Elements)')
ax2.set_ylabel('Speedup (Custom/Thrust)')
ax2.set_title('Performance Speedup')
ax2.set_xscale('log')
ax2.grid(True)

# Plot 3: Throughput
ax3.plot(df['Elements'], df['Throughput_M_per_sec'], 'mo-')
ax3.set_xlabel('Dataset Size (Elements)')
ax3.set_ylabel('Throughput (M elements/sec)')
ax3.set_title('Processing Throughput')
ax3.set_xscale('log')
ax3.grid(True)

# Plot 4: Efficiency (Throughput per GHz - if we had frequency data)
elements_per_gb = df['Elements'] / (df['Elements'] * 4 / 1024**3)  # Rough approximation
ax4.plot(df['Elements'], elements_per_gb, 'co-')
ax4.set_xlabel('Dataset Size (Elements)')
ax4.set_ylabel('Elements per GB Memory')
ax4.set_title('Memory Efficiency')
ax4.set_xscale('log')
ax4.grid(True)

plt.tight_layout()
plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
plt.show()
print("Plot saved as benchmark_results.png")
EOF

    python3 "$plot_script" "$csv_file"
    rm -f "$plot_script"
    log_info "Plots generated and saved as benchmark_results.png"
}

# ================================================================
# MAIN SCRIPT LOGIC
# ================================================================

main() {
    case "$1" in
        "help"|"-h"|"--help"|"")
            print_header
            print_usage
            ;;
        "examples")
            print_header
            print_examples
            ;;
        "check")
            print_header
            check_system
            ;;
        "setup")
            print_header
            setup_project
            ;;
        "build")
            print_header
            build_program
            ;;
        "run")
            print_header
            run_with_size "$2"
            ;;
        "test")
            print_header
            run_tests
            ;;
        "benchmark")
            print_header
            run_benchmark
            ;;
        "monitor")
            monitor_gpu
            ;;
        "clean")
            print_header
            clean_files
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            echo ""
            print_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
