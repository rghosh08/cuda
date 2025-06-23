# Make executable and setup
chmod +x mapreduce.sh
./mapreduce.sh setup

# Get help and examples  
./mapreduce.sh help
./mapreduce.sh examples

# Check your system
./mapreduce.sh check

# Run tests
./mapreduce.sh run                    # Default 10M elements
./mapreduce.sh run 1000000           # 1M elements
./mapreduce.sh test                  # Full test suite
./mapreduce.sh benchmark             # Performance testing

# Monitor GPU while running
./mapreduce.sh monitor
