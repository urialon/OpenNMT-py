val_source=../synpos/data/java-small-processed/val_src.txt
test_source=../synpos/data/java-small-processed/test_src.txt
val_target=../synpos/data/java-seq2seq-data/data.TargetType.seq.val.target.txt
test_target=../synpos/data/java-seq2seq-data/data.TargetType.seq.test.target.txt

model_name=$1
output_dir=${model_name}/validation_results/

#set -o xtrace

mkdir -p ${output_dir}

for model in $(ls -tr ${model_name}/model_step_*.pt)
do
    echo Translating ${model}...
    output=${output_dir}/$(basename $model).txt
    python onmt/bin/translate.py -model ${model} -src ${val_source} -output ${output} -n_best 5 -beam_size 5 -gpu 0
done

for translation in $(ls ${output_dir}/model*); do
    echo $translation;
    python eval_seq2seq.py --expected ${val_target} --actual ${translation}; done
for translation in $(ls ${output_dir}/model*); do
    echo $translation;
    python eval_seq2seq.py --expected ${val_target} --actual ${translation}; done > ${output_dir}/validation_log.txt

best_model=$(basename $(cat ${output_dir}/validation_log.txt | grep 'Final' | sort -n -k 3 | tail -1 | cut -d' ' -f2))
best_model=${best_model%.*}

echo
echo Best model: ${best_model}
echo
echo Best model validation scores:
python eval_seq2seq.py --expected ${val_target} --actual ${output_dir}/${best_model}.txt

echo
echo Test results:
test_output=${model_name}/test_translation_${best_model}
python onmt/bin/translate.py -model ${model_name}/${best_model} -src ${test_source} -output ${test_output} -n_best 5 -beam_size 5 -gpu 0
python eval_seq2seq.py --expected ${test_target} --actual ${test_output}
python eval_seq2seq.py --expected ${test_target} --actual ${test_output} > ${model_name}/results_${best_model}
