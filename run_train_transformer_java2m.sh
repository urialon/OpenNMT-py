#!/usr/bin/env bash

set -e

model_name=$1

#mkdir ${model_name}
python -u onmt/bin/train.py --config java_med2m_nofeat.yml --save_mode ${model_name}/model ${@:2}

notify-run send 'Finished training'

