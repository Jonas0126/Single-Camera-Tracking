
buffer_size=(10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
threshold=(0.5)
lambda=(0.8)

for ((i=0; i < ${#buffer_size[@]}; i++))
do
    for ((j=0; j < ${#threshold[@]}; j++))
    do
        for ((k=0; k < ${#lambda[@]}; k++))
        do
        echo python main.py -f ../REID/IMAGE/0902_150000_151900/ -l ../REID/LABEL/0902_150000_151900/ --out result_label/0902_150000_151900_re_${buffer_size[i]}_${threshold[j]}_${lambda[k]} --buffer_size ${buffer_size[i]} --threshold ${threshold[j]} --lambda_value ${lambda[k]} 
        python main.py -f ../REID/IMAGE/0902_150000_151900/ -l ../REID/LABEL/0902_150000_151900/ --out result_label/0902_150000_151900_re_${buffer_size[i]}_${threshold[j]}_${lambda[k]} --buffer_size ${buffer_size[i]} --threshold ${threshold[j]} --lambda_value ${lambda[k]}
        
        done

    done

done

