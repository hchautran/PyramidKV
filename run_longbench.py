import os
import json
import random
import argparse

import numpy as np
from tqdm import tqdm
import torch
from transformers import LlamaForCausalLM , AutoTokenizer
from datasets import load_dataset
from utils import LongBench, set_seed, model2maxlen, model2prompt, dataset2maxlen




def main(args):
    
    for key in model2maxlen:
        if key in args.model_path.lower():
            model_max_len = model2maxlen[key]

    print("Finish loading model and tokenizer")
    
    model_name = args.model_path.split("/")[-1].lower()

    os.makedirs(os.path.join(args.save_dir, f"{model_name}", args.dataset), exist_ok=True)

    fout = open(os.path.join(args.save_dir, f"{model_name}", args.dataset, f"{args.method}.json"), "w")
    dataset = LongBench(args)
    

    for i in tqdm(range(10)):
        sample = dataset[i]
        tokenized_prompts = tokenizer(sample['prompt'], padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        if len(batch_input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
            
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids

        context_length = batch_input_ids.shape[-1]
                
        output = model.generate(
            **tokenized_prompts,
            output_attentions = args.output_attentions,
            max_new_tokens=dataset2maxlen[args.dataset],
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            min_length=context_length+1,
            eos_token_id=[tokenizer.eos_token_id]
        )

        outputs = tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)
        
        sample['pred'] = outputs[0]
        # print(f"debbug batch_outputs {batch_outputs}")
        torch.cuda.empty_cache()
        fout.write(json.dumps(sample) + "\n")
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    
    args = parser.parse_args()

    
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )
    if args.method.lower() == 'pyramidkv':
        from models.llama.pyramidkv  import LlamaForCausalLM
    elif args.method.lower() == 'pitomekv':
        from models.llama.pitomekv import LlamaForCausalLM
    else:
        from transformers import LlamaForCausalLM

    print('using:',args.model_path, 'with', args.method.lower(), 'on', args.dataset)
    
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation,
    )
        

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    save_dir = args.save_dir
    
    max_capacity_prompts = args.max_capacity_prompts

    main(args)
