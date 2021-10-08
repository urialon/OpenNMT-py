val_source=../synpos/data/java-small-processed/val_src.txt
test_source=../synpos/data/java-small-processed/test_src.txt
val_target=../synpos/data/java-seq2seq-data/data.TargetType.seq.val.target.txt
test_target=../synpos/data/java-seq2seq-data/data.TargetType.seq.test.target.txt

model_name=$1
output_dir=${model_name}/validation_results/
val_src_feats="{'feat_0': '../synpos/data/java-small-processed/val_feat0.txt', 'feat_1': '../synpos/data/java-small-processed/val_feat1.txt', 'feat_2': '../synpos/data/java-small-processed/val_feat2.txt', 'feat_3': '../synpos/data/java-small-processed/val_feat3.txt', 'feat_4': '../synpos/data/java-small-processed/val_feat4.txt', 'feat_5': '../synpos/data/java-small-processed/val_feat5.txt', 'feat_6': '../synpos/data/java-small-processed/val_feat6.txt', 'feat_7': '../synpos/data/java-small-processed/val_feat7.txt', 'feat_8': '../synpos/data/java-small-processed/val_feat8.txt', 'feat_9': '../synpos/data/java-small-processed/val_feat9.txt', 'feat_10': '../synpos/data/java-small-processed/val_feat10.txt', 'feat_11': '../synpos/data/java-small-processed/val_feat11.txt', 'feat_12': '../synpos/data/java-small-processed/val_feat12.txt', 'feat_13': '../synpos/data/java-small-processed/val_feat13.txt', 'feat_14': '../synpos/data/java-small-processed/val_feat14.txt', 'feat_15': '../synpos/data/java-small-processed/val_feat15.txt', 'feat_16': '../synpos/data/java-small-processed/val_feat16.txt', 'feat_17': '../synpos/data/java-small-processed/val_feat17.txt', 'feat_18': '../synpos/data/java-small-processed/val_feat18.txt', 'feat_19': '../synpos/data/java-small-processed/val_feat19.txt', 'feat_20': '../synpos/data/java-small-processed/val_feat20.txt', 'feat_21': '../synpos/data/java-small-processed/val_feat21.txt', 'feat_22': '../synpos/data/java-small-processed/val_feat22.txt', 'feat_23': '../synpos/data/java-small-processed/val_feat23.txt'}"
test_src_feats="{'feat_0': '../synpos/data/java-small-processed/test_feat0.txt', 'feat_1': '../synpos/data/java-small-processed/test_feat1.txt', 'feat_2': '../synpos/data/java-small-processed/test_feat2.txt', 'feat_3': '../synpos/data/java-small-processed/test_feat3.txt', 'feat_4': '../synpos/data/java-small-processed/test_feat4.txt', 'feat_5': '../synpos/data/java-small-processed/test_feat5.txt', 'feat_6': '../synpos/data/java-small-processed/test_feat6.txt', 'feat_7': '../synpos/data/java-small-processed/test_feat7.txt', 'feat_8': '../synpos/data/java-small-processed/test_feat8.txt', 'feat_9': '../synpos/data/java-small-processed/test_feat9.txt', 'feat_10': '../synpos/data/java-small-processed/test_feat10.txt', 'feat_11': '../synpos/data/java-small-processed/test_feat11.txt', 'feat_12': '../synpos/data/java-small-processed/test_feat12.txt', 'feat_13': '../synpos/data/java-small-processed/test_feat13.txt', 'feat_14': '../synpos/data/java-small-processed/test_feat14.txt', 'feat_15': '../synpos/data/java-small-processed/test_feat15.txt', 'feat_16': '../synpos/data/java-small-processed/test_feat16.txt', 'feat_17': '../synpos/data/java-small-processed/test_feat17.txt', 'feat_18': '../synpos/data/java-small-processed/test_feat18.txt', 'feat_19': '../synpos/data/java-small-processed/test_feat19.txt', 'feat_20': '../synpos/data/java-small-processed/test_feat20.txt', 'feat_21': '../synpos/data/java-small-processed/test_feat21.txt', 'feat_22': '../synpos/data/java-small-processed/test_feat22.txt', 'feat_23': '../synpos/data/java-small-processed/test_feat23.txt'}"

#set -o xtrace

mkdir -p ${output_dir}

for model in $(ls -tr ${model_name}/model*.pt)
do
    echo Translating ${model}...
    output=${output_dir}/$(basename $model).txt
    if [ -f $output ]; then
       echo "Translation for $output exists."
    else
       python onmt/bin/translate.py -model ${model} -src ${val_source} -output ${output} -n_best 5 -beam_size 5 -gpu 0 -src_feats "${val_src_feats}"
    fi
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
python onmt/bin/translate.py -model ${model_name}/${best_model} -src ${test_source} -output ${test_output} -n_best 5 -beam_size 5 -gpu 0 -src_feats "${test_src_feats}" -batch_size 16
python eval_seq2seq.py --expected ${test_target} --actual ${test_output}
python eval_seq2seq.py --expected ${test_target} --actual ${test_output} > ${model_name}/results_${best_model}
