val_source=../synpos/data/java-small-processed/val_src.txt
test_source=../synpos/data/java-small-processed/test_src.txt
val_target=../synpos/data/java-seq2seq-data/data.TargetType.seq.val.target.txt
test_target=../synpos/data/java-seq2seq-data/data.TargetType.seq.test.target.txt

model_name=java_small_transformer_nofeat
output_dir=${model_name}/validation_results/

#set -o xtrace

mkdir -p ${output_dir}

for model in $(ls -t ${model_name}/model_step_*.pt)
do
    output=${output_dir}/$(basename $model).txt
    python onmt/bin/translate.py -model ${model} -src ${val_source} -output ${output} -n_best 5 -beam_size 5 -gpu 0
done

for translation in $(ls ${output_dir}/model*); do
    echo $translation;
    python eval_seq2seq.py --expected ${val_target} --actual ${translation}; done
for translation in $(ls ${output_dir}/model*); do
    echo $translation;
    python eval_seq2seq.py --expected ${val_target} --actual ${translation}; done > ${output_dir}/validation_log.txt

best_model=$(basename $(cat ${output_dir}/validation_log.txt | grep 'Final' | sort -n -k 3 | tail -1 | cut -d' ' -f2)%.*)
best=${best%.*}

echo Best model: ${best}
test_output=${model_name}/test_translation_${best}
python onmt/bin/translate.py -model ${model_name}/${model} -src ${test_source} -output ${test_output} -n_best 5 -beam_size 5 -gpu 0
python eval_seq2seq.py --expected ${test_target} --actual ${test_output}; done
python eval_seq2seq.py --expected ${test_target} --actual ${test_output}; done > ${model_name}/results_${best}
