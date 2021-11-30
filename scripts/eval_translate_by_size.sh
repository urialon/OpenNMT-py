for test_size in 400 600 2000; do
dataset=java-med-test
test_source=../synpos/data/${dataset}-processed/test${test_size}_src.txt
test_target=../synpos/data/${dataset}/java-med.TargetType.seq.test${test_size}-sampled.target.txt
model_name=java_med_nofeat
best_model=model_step_980000.pt
test_output=${model_name}/test${test_size}_translation_${best_model}
python onmt/bin/translate.py -model ${model_name}/${best_model} -src ${test_source} -output ${test_output} -n_best 5 -beam_size 5 -gpu 0 -batch_size 32 --replace_unk

python eval_seq2seq.py --expected ${test_target} --actual ${test_output}
python eval_seq2seq.py --expected ${test_target} --actual ${test_output} > ${model_name}/results${test_size}_${best_model}

done