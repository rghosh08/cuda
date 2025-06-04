# Install gnuplot
sudo apt install gnuplot

# Run your program and plot
./cuda_nn_loss_temp > training.log
grep "^Epoch" training.log | awk '{print $2, $5, $12}' > result_temp.txt

# Plot in terminal
gnuplot -e "set terminal dumb 120 30; plot 'result_temp.txt' using 1:2 with lines title 'Loss', '' using 1:3 with lines title 'Temp'"

# Temperature bar chart
echo "Temperature progression:"
grep '^Epoch' log.txt | awk 'NR%50==0 || NR==1 {
    gsub(/°C/,"",$11)
    temp = $11
    printf "%4d | ", $2
    for(i=0; i<(temp-40); i++) printf "█"
    printf " %d°C\n", temp
}'
