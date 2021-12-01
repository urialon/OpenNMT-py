dataset=java-med-test
model_name=java_med_sharemlp_rtx8000
best_model=model_step_980000.pt

test_size=400
test_source=../synpos/data/${dataset}-processed/test${test_size}_src.txt
test_target=../synpos/data/${dataset}/java-med.TargetType.seq.test${test_size}-sampled.target.txt
test_src_feats="{'feat_0': '../synpos/data/${dataset}-processed/test${test_size}_feat0.txt', 'feat_1': '../synpos/data/${dataset}-processed/test${test_size}_feat1.txt', 'feat_2': '../synpos/data/${dataset}-processed/test${test_size}_feat2.txt', 'feat_3': '../synpos/data/${dataset}-processed/test${test_size}_feat3.txt', 'feat_4': '../synpos/data/${dataset}-processed/test${test_size}_feat4.txt', 'feat_5': '../synpos/data/${dataset}-processed/test${test_size}_feat5.txt', 'feat_6': '../synpos/data/${dataset}-processed/test${test_size}_feat6.txt', 'feat_7': '../synpos/data/${dataset}-processed/test${test_size}_feat7.txt', 'feat_8': '../synpos/data/${dataset}-processed/test${test_size}_feat8.txt', 'feat_9': '../synpos/data/${dataset}-processed/test${test_size}_feat9.txt', 'feat_10': '../synpos/data/${dataset}-processed/test${test_size}_feat10.txt', 'feat_11': '../synpos/data/${dataset}-processed/test${test_size}_feat11.txt', 'feat_12': '../synpos/data/${dataset}-processed/test${test_size}_feat12.txt', 'feat_13': '../synpos/data/${dataset}-processed/test${test_size}_feat13.txt', 'feat_14': '../synpos/data/${dataset}-processed/test${test_size}_feat14.txt', 'feat_15': '../synpos/data/${dataset}-processed/test${test_size}_feat15.txt', 'feat_16': '../synpos/data/${dataset}-processed/test${test_size}_feat16.txt', 'feat_17': '../synpos/data/${dataset}-processed/test${test_size}_feat17.txt', 'feat_18': '../synpos/data/${dataset}-processed/test${test_size}_feat18.txt', 'feat_19': '../synpos/data/${dataset}-processed/test${test_size}_feat19.txt', 'feat_20': '../synpos/data/${dataset}-processed/test${test_size}_feat20.txt', 'feat_21': '../synpos/data/${dataset}-processed/test${test_size}_feat21.txt', 'feat_22': '../synpos/data/${dataset}-processed/test${test_size}_feat22.txt', 'feat_23': '../synpos/data/${dataset}-processed/test${test_size}_feat23.txt'}"
test_output=${model_name}/test${test_size}_translation_${best_model}
python onmt/bin/translate.py -model ${model_name}/${best_model} -src ${test_source} -output ${test_output} -n_best 5 -beam_size 5 -gpu 0 --replace_unk -src_feats "${test_src_feats}" -batch_size 16

python eval_seq2seq.py --expected ${test_target} --actual ${test_output}
python eval_seq2seq.py --expected ${test_target} --actual ${test_output} > ${model_name}/results${test_size}_${best_model}

test_size=600
test_source=../synpos/data/${dataset}-processed/test${test_size}_src.txt
test_target=../synpos/data/${dataset}/java-med.TargetType.seq.test${test_size}.target.txt
test_output=${model_name}/test${test_size}_translation_${best_model}
python onmt/bin/translate.py -model ${model_name}/${best_model} -src ${test_source} -output ${test_output} -n_best 5 -beam_size 5 -gpu 0 --replace_unk -src_feats "${test_src_feats}" -batch_size 16

python eval_seq2seq.py --expected ${test_target} --actual ${test_output}
python eval_seq2seq.py --expected ${test_target} --actual ${test_output} > ${model_name}/results${test_size}_${best_model}

test_size=2000
test_source=../synpos/data/${dataset}-processed/test${test_size}_src.txt
test_target=../synpos/data/${dataset}/java-med.TargetType.seq.test${test_size}.target.txt
test_output=${model_name}/test${test_size}_translation_${best_model}
python onmt/bin/translate.py -model ${model_name}/${best_model} -src ${test_source} -output ${test_output} -n_best 5 -beam_size 5 -gpu 0 --replace_unk -src_feats "${test_src_feats}" -batch_size 16

python eval_seq2seq.py --expected ${test_target} --actual ${test_output}
python eval_seq2seq.py --expected ${test_target} --actual ${test_output} > ${model_name}/results${test_size}_${best_model}