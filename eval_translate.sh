validation_content=/scratch/urialon/gen/data/seq2seq-exp-types-func20/data.TargetType.seq.val.source.txt
validation_names=/scratch/urialon/gen/data/seq2seq-exp-types-func20/data.TargetType.seq.val.target.txt
#test_content=java-if-seq2seq/raw/data.TargetType.seq.val.source.txt
#test_names=java-if-seq2seq/raw/data.TargetType.seq.test.target.txt
model_name=java-exp-types-func20/
output_dir=${model_name}/validation_results

mkdir -p ${output_dir}/${model_name}

for model in $(ls -t ${model_name}/model_step_*.pt)
do
    output=${output_dir}/${model}.txt
    python3 translate.py -model ${model} -src ${validation_content} -output ${output} -batch_size 128 -n_best 5 -beam_size 5
    #paste -d"," ${validation_names} ${output} > $(dirname $output)/combined_$(basename $output)
done

for translation in $(ls ${output_dir}/${model_name}/model*); do echo $translation; python ../gen/scripts/eval_seq2seq.py --expected ${validation_names} --actual ${translation}; done
for translation in $(ls ${output_dir}/${model_name}/model*); do echo $translation; python ../gen/scripts/eval_seq2seq.py --expected ${validation_names} --actual ${translation}; done > ${output_dir}/validation_log.txt
