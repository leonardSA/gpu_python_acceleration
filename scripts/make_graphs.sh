#!/usr/bin/env bash

if [ $# -ne 5 ] && [ $# -ne 6 ]; then
    echo -e "USAGE:\t$0 python3.XX executable start step stop [naive]"
    echo -e "ARGUMENTS:"
    echo -e "\tpython3.XX:\tpython interpreter"
    echo -e "\texecutable:\tpython code to execute with the interpreter"
    echo -e "\tstart:\t\tstart size of matrices such as AxB=C with A of shape" \
        "(start, start) and B of shape (start, start)"
    echo -e "\tstep:\t\tstarting size increments with step"
    echo -e "\tstop:\t\tend size of matrices such as AxB=C with A of shape" \
        "(stop, stop) and B of shape (stop, stop)"
    echo -e "\tnaive:\t\tuse naive implementation"
    exit 1
fi

PY_VERSION=$1
EXEC=$2
START=$3
STEP=$4
STOP=$5
if [[ $6 != "" ]]; then
    NAIVE_ARG="--naive"
    NAIVE="naive"
fi

# Create data files

buffer_data="buffer_transfer.dat"
execution_data="execution_time.dat"
accuracy_data="float_accuracy.dat"

if [[ $NAIVE != "" ]]; then
    buffer_data=$NAIVE"_"$buffer_data
    execution_data=$NAIVE"_"$execution_data
    accuracy_data=$NAIVE"_"$accuracy_data
fi

if [ -f $buffer_data ]; then rm $buffer_data ; fi
if [ -f $execution_data ]; then rm $execution_data ; fi
if [ -f $accuracy_data ]; then rm $accuracy_data ; fi

echo -e "# Matrix size (nxn)\tCopy host to gpu\tCopy gpu to host" >> $buffer_data
echo -e "# Matrix size (nxn)\tGPU execution time\tNumpy matmul execution time" >> $execution_data
echo -e "# Matrix size (nxn)\tLower bound diff\tHigher bound diff" >> $accuracy_data

# Generate data

for i in $(seq $START $STEP $STOP); do
    cmd="$PY_VERSION $EXEC $i $i $i $i -t -p -n $NAIVE_ARG"
    echo "Executing $cmd"
    output=$($cmd)
    matrix_nb_elements=$(( i * i ))
    declare -a ARRAY
    ARRAY=($output)
    echo -e "$matrix_nb_elements\t${ARRAY[0]}\t${ARRAY[2]}" >> $buffer_data
    echo -e "$matrix_nb_elements\t${ARRAY[1]}\t${ARRAY[3]}" >> $execution_data
    echo -e "$matrix_nb_elements\t${ARRAY[4]}\t${ARRAY[5]}" >> $accuracy_data
done

# Create plot scripts

gnuplot_plot_buffer=${buffer_data%.*}".plot"
gnuplot_plot_execution=${execution_data%.*}".plot"
gnuplot_plot_accuracy=${accuracy_data%.*}".plot"

if [ -f $gnuplot_plot_buffer ]; then rm $gnuplot_plot_buffer ; fi
if [ -f $gnuplot_plot_execution ]; then rm $gnuplot_plot_execution ; fi
if [ -f $gnuplot_plot_accuracy ]; then rm $gnuplot_plot_accuracy ; fi

cp gnuplot_plot_canvas $gnuplot_plot_buffer 
cp gnuplot_plot_canvas $gnuplot_plot_execution
cp gnuplot_plot_canvas $gnuplot_plot_accuracy

chmod a+x $gnuplot_plot_buffer $gnuplot_plot_execution $gnuplot_plot_accuracy

# Plot data

# Plot buffer times

sed -i "s/DATAFILE/$buffer_data/g" $gnuplot_plot_buffer
sed -i "s/FILENAME/${buffer_data%.*}.svg/g" $gnuplot_plot_buffer
sed -i "s/TITLE1/Buffer Transfers/g" $gnuplot_plot_buffer
sed -i "s/TITLE2/Copying onto GPU/g" $gnuplot_plot_buffer
sed -i "s/TITLE3/Copying from GPU/g" $gnuplot_plot_buffer
sed -i "s/LABELY/Time in seconds/g" $gnuplot_plot_buffer

./$gnuplot_plot_buffer

# Plot execution times

sed -i "s/DATAFILE/$execution_data/g" $gnuplot_plot_execution
sed -i "s/FILENAME/${execution_data%.*}.svg/g" $gnuplot_plot_execution
sed -i "s/TITLE1/Execution Time/g" $gnuplot_plot_execution
sed -i "s/TITLE2/GPU execution/g" $gnuplot_plot_execution
sed -i "s/TITLE3/Numpy matmul execution/g" $gnuplot_plot_execution
sed -i "s/LABELY/Time in seconds/g" $gnuplot_plot_execution

./$gnuplot_plot_execution

# Plot accuracy

sed -i "s/DATAFILE/$accuracy_data/g" $gnuplot_plot_accuracy
sed -i "s/FILENAME/${accuracy_data%.*}.svg/g" $gnuplot_plot_accuracy
sed -i "s/TITLE1/Accuracy differences between matrices/g" $gnuplot_plot_accuracy
sed -i "s/TITLE2/Accuracy lower bound/g" $gnuplot_plot_accuracy
sed -i "s/TITLE3/Accuracy higher bound/g" $gnuplot_plot_accuracy
sed -i "s/LABELY/ /g" $gnuplot_plot_accuracy

./$gnuplot_plot_accuracy

# Clean up
# rm *.plot
# rm *.dat
