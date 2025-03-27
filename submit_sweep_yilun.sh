#!/bin/bash

list_of_token_mixing_struct=("bilinearbtt" "low_rank")
list_of_learning_rate=(0.0005 0.0001 0.00005 0.00001)
# list_of_learning_rate=(0.0005)


for token_mixing_struct in "${list_of_token_mixing_struct[@]}"; do 
    for learning_rate in "${list_of_learning_rate[@]}"; do 
        echo "token_mixing_struct: $token_mixing_struct, learning_rate: $learning_rate"
        sbatch submit_batch_yilun.sh $token_mixing_struct $learning_rate
    done
done
