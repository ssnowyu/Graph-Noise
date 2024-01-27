#!/bin/bash
CUDA=$1
SEED_ARRAY=(0 1 2 3 4 5 6 7 8 9)
DATASEED_ARRAY=(0 1 2 3 4 5 6 7 8 9)
NOISE_RATE_ARRAY=(0.6 0.7 0.8 0.9 1.0)
SIGMA_ARRAY=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)

for i in {0..4}; do
    for noise_rate in "${NOISE_RATE_ARRAY[@]}"; do
        for sigma in "${SIGMA_ARRAY[@]}"; do
            seed=${SEED_ARRAY[i]}
            options="-m seed=${seed} experiment=cora/gcn data.noise_rate=${noise_rate} data.sigma=${sigma}"
            cmd="CUDA_VISIBLE_DEVICES=${CUDA} python src/train.py ${options}"
            echo $cmd
            eval $cmd
        done
    done
done

# export CUDA_VISIBLE_DEVICES=1

# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.3 data.sigma=0.3
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.3 data.sigma=0.2
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.3 data.sigma=0.1
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.4 data.sigma=0.3
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.4 data.sigma=0.2
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.4 data.sigma=0.1
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.5 data.sigma=0.3
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.5 data.sigma=0.2
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.5 data.sigma=0.1
# # python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.6 data.sigma=0.3
# # python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.6 data.sigma=0.2
# # python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.6 data.sigma=0.1
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.7 data.sigma=0.3
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.7 data.sigma=0.2
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.7 data.sigma=0.1
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.8 data.sigma=0.3
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.8 data.sigma=0.2
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.8 data.sigma=0.1
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.9 data.sigma=0.3
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.9 data.sigma=0.2
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=0.9 data.sigma=0.1
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=1 data.sigma=0.3
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=1 data.sigma=0.2
# python src/train.py -m seed=0,1,2 experiment=cora/gcn data.noise_rate=1 data.sigma=0.1
