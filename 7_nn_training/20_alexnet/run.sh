# Minimal monitoring - just temperature and power
echo "🚀 Starting..."; nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits --loop=0.1 > temp_log_1000.csv & P=$!; time ./alexnet_cu; kill $P; echo "📊 Max temp: $(cut -d',' -f1 temp_log.csv | sort -n | tail -1)°C, Max power: $(cut -d',' -f2 temp_log.csv | sort -n | tail -1)W"
