#!/usr/bin/env bash

if [ $# -ne 2 ]; then
    echo "USAGE: $0 python3.XX executable"
    exit 1
fi

PY_VERSION=$1
EXEC=$2

# Create data files

buffer_data="buffer_transfer.dat"
execution_data="execution_time.dat"
precision_data="float_precision.dat"

if [ -f $buffer_data ]; then rm $buffer_data ; fi
if [ -f $execution_data ]; then rm $execution_data ; fi
if [ -f $precision_data ]; then rm $precision_data ; fi

echo -e "# Matrix size (nxn)\tCopy host to gpu\tCopy gpu to host" >> $buffer_data
echo -e "# Matrix size (nxn)\tGPU execution time\tNumpy matmul execution time" >> $execution_data
echo -e "# Matrix size (nxn)\tLower bound diff\tHigher bound diff" >> $precision_data

# Generate data

for i in $(seq 10 10 1500); do
    echo "Executing $PY_VERSION $EXEC $i $i $i $i -t -p -n"
    output=$($PY_VERSION $EXEC $i $i $i $i -t -p -n)
    matrix_nb_elements=$(( i * i ))
    declare -a ARRAY
    ARRAY=($output)
    echo -e "$matrix_nb_elements\t${ARRAY[0]}\t${ARRAY[2]}" >> $buffer_data
    echo -e "$matrix_nb_elements\t${ARRAY[1]}\t${ARRAY[3]}" >> $execution_data
    echo -e "$matrix_nb_elements\t${ARRAY[4]}\t${ARRAY[5]}" >> $precision_data
done

# Create plot scripts

gnuplot_plot_buffer=${buffer_data%.*}".plot"
gnuplot_plot_execution=${execution_data%.*}".plot"
gnuplot_plot_precision=${precision_data%.*}".plot"

if [ -f $gnuplot_plot_buffer ]; then rm $gnuplot_plot_buffer ; fi
if [ -f $gnuplot_plot_execution ]; then rm $gnuplot_plot_execution ; fi
if [ -f $gnuplot_plot_precision ]; then rm $gnuplot_plot_precision ; fi

cp gnuplot_plot_canvas $gnuplot_plot_buffer 
cp gnuplot_plot_canvas $gnuplot_plot_execution
cp gnuplot_plot_canvas $gnuplot_plot_precision

chmod a+x $gnuplot_plot_buffer $gnuplot_plot_execution $gnuplot_plot_precision

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

# Plot precision

sed -i "s/DATAFILE/$precision_data/g" $gnuplot_plot_precision
sed -i "s/FILENAME/${precision_data%.*}.svg/g" $gnuplot_plot_precision
sed -i "s/TITLE1/Precision differences between matrices/g" $gnuplot_plot_precision
sed -i "s/TITLE2/Precision lower bound/g" $gnuplot_plot_precision
sed -i "s/TITLE3/Precision higher bound/g" $gnuplot_plot_precision
sed -i "s/LABELY/ /g" $gnuplot_plot_precision

./$gnuplot_plot_precision

# Clean up
# rm *.plot
# rm *.dat
