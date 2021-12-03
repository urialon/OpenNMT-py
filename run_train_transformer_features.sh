#!/usr/bin/env bash

set -e

model_name=$1
#mkdir ${model_name}

#source activate 3090-local

python -u onmt/bin/train.py --config java_med_features.yaml --save_mode ${model_name}/model ${@:2}
#python -u onmt/bin/train.py --config java_small_features_shared_vocab.yaml --save_mode ${model_name}/model ${@:2}

notify-run send 'Finished training transformer'

