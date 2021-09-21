val_source=
val_target=

model_name=
output_dir=${model_name}/validation_results/

mkdir -p ${output_dir}

for model in $(ls -t ${model_name}/model_step_*.pt)
do
    output=${output_dir}/${model}.txt
    python3 python onmt/bin/translate.py -model ${model} -src ${val_target} -output ${output} -n_best 5 -beam_size 5 -gpu 0
done

for translation in $(ls ${output_dir}/model*); do
    echo $translation;
    python eval_seq2seq.py --expected ${val_target} --actual ${translation}; done
for translation in $(ls ${output_dir}/model*); do
    echo $translation;
    python eval_seq2seq.py --expected ${val_target} --actual ${translation}; done > ${output_dir}/validation_log.txt
