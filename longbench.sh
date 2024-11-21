export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

max_capacity_prompts=64 # 128,2048 in paper
attn_implementation="eager" # Support "flash_attention_2", "sdpa", "eager".
model_path="meta-llama/Meta-Llama-3-8B-Instruct"
save_dir=${source_path}"results_long_bench" # path to result save_dir
results_dir=results_long_bench/meta-llama-3-8b-instruct


# datasets = []
for dataset in "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "gov_report" "multi_news" "trec" "triviaqa" "samsum" "passage_count" "passage_retrieval_en" "lcc" "repobench-p"
do
    for method in PiToMeKV FullKV PyramidKV
    do
        python run_longbench.py \
            --method ${method} \
            --model_path ${model_path} \
            --max_capacity_prompts ${max_capacity_prompts} \
            --attn_implementation ${attn_implementation} \
            --save_dir ${save_dir} \
            --use_cache True \
            --dataset ${dataset} 

        python eval.py \
            --results_dir ${results_dir}
    done
done