#!/bin/bash
CUDA=$1
SEED_ARRAY=(0 1 2 3 4 5 6 7 8 9)
DATASEED_ARRAY=(0 1 2 3 4 5 6 7 8 9)
FEAT_NOISE_RATE_ARRAY=(1.0)
FEAT_SIGMA_ARRAY=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)
NOISE_TYPE_ARRAY=(none feat missing-edge redundant-edge error-edge)
EDGE_RATE_ARRAY=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for i in {0..4}; do
    for feat_noise_rate in "${FEAT_NOISE_RATE_ARRAY[@]}"; do
        for feat_sigma in "${FEAT_SIGMA_ARRAY[@]}"; do
            seed=${SEED_ARRAY[i]}
            options="-m seed=${seed} experiment=cora/gcn data.noise_type=feat data.feat_noise_rate=${feat_noise_rate} data.feat_sigma=${feat_sigma}"
            cmd="CUDA_VISIBLE_DEVICES=${CUDA} python src/train.py ${options}"
            echo $cmd
            eval $cmd
        done
    done
done

for i in {0..4}; do
    for j in {2..4}; do
        for edge_rate in "${EDGE_RATE_ARRAY[@]}"; do
            noise_type=${NOISE_TYPE_ARRAY[j]}
            seed=${SEED_ARRAY[i]}
            options="-m seed=${seed} experiment=cora/gcn data.noise_type=${noise_type} data.edge_rate=${edge_rate}"
            cmd="CUDA_VISIBLE_DEVICES=${CUDA} python src/train.py ${options}"
            echo $cmd
            eval $cmd
        done
    done
done
