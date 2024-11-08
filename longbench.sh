export CUDA_VISIBLE_DEVICES=1,2,3,4,5

method=PiToMeKV # Support PyramidKV, SnapKV, H2O, StreamingLLM
# method=PyramidKV # Support PyramidKV, SnapKV, H2O, StreamingLLM
max_capacity_prompts=64 # 128,2048 in paper
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "eager".
model_path="meta-llama/Llama-2-7b-hf"
save_dir=${source_path}"results_long_bench" # path to result save_dir



python run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True
