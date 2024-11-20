
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.cache import PiToMeCache
from models.llama.pyramidkv import LlamaForCausalLM as PyramidLlamaForCausalLM 
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import wandb
from const import  (
   LLAMA2_7B,
   LLAMA3_8B,
   LLAMA3_1_8B,
   LLAMA3_2_3B,
   LLAMA3_2_1B
)
import os


model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

model2maxlen = {
   "llama2": 3950,
   "llama-2": 3950,
   "llama3": 7950,
   "llama-3": 7950,
   "mistral": 31500
}

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}


# model_ckt = LLAMA3_1_8B 




def build_chat(prompt):
        prompt = f"[INST] {prompt} [/INST]"
        return prompt



def cal_energy(metric:torch.Tensor, sigma:float=0.1):
   metric = F.normalize(metric, p=2, dim=-1) 
   sim = metric@metric.transpose(-1,-2)
   energy_score = (torch.exp(-(((1 - sim)/sigma)**2 * 0.5))).mean(-1) *  1/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))
   # energy_score = F.elu(sim - sigma).mean(-1)

   return energy_score


def manual_infer_with_llama_with_attention(prompt, max_length=50):
   # past_key_values = PiToMeCache()

   input_ids = tokenizer.encode(prompt, return_tensors='pt')
   all_layers_attentions = [] 

   for _ in range(max_length):

      raw_outputs = model(input_ids, output_attentions=True, return_dict=True, use_cache=True)
      output = raw_outputs.logits
      next_token_logits = output[:, -1, :]
      
      attentions = raw_outputs.attentions

      next_token = torch.argmax(next_token_logits, dim=-1)

      input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

      if next_token in tokenizer.all_special_ids:
         break
      
   for i in range(len(attentions)):
      all_layers_attentions.append(attentions[i].detach().cpu())
   
   return tokenizer.decode(input_ids[0], skip_special_tokens=True), input_ids[0], all_layers_attentions, raw_outputs.past_key_values



def plot_attention_energy_map(attention, energy):
      fig = plt.figure(figsize=(9, 10))
      gs = gridspec.GridSpec(9, 10)

      ax1 = fig.add_subplot(gs[:-1, :])
      ax2 = fig.add_subplot(gs[-1, :], sharex=ax1)

      ax1.imshow(attention, vmax=100)
      # ax2.imshow(attention_mean.reshape(1, -1)*100)
      ax2.imshow(energy.mean(1).reshape(1, -1), cmap='inferno', aspect='auto')
      # ax3.imshow(energy_value.reshape(1, -1), cmap='inferno', aspect='auto')
      plt.setp(ax1.get_xticklabels(), visible=False)

      plt.tight_layout()
      plt.savefig(f'{directory}/layer{layer_idx}.png', dpi=300, format='png')
      plt.show()


if __name__ == '__main__':
   model_ckt = LLAMA3_8B 
   model_ckt = LLAMA2_7B 
   tokenizer = AutoTokenizer.from_pretrained(model_ckt)
   model = PyramidLlamaForCausalLM.from_pretrained(
      model_ckt,
      torch_dtype=torch.float16,
      low_cpu_mem_usage=True,
      device_map="auto",
      use_cache=True,
   )
   dataset = 'multi_news'
   longbench = load_dataset('THUDM/LongBench', dataset, split='test')
   longbench_filtered = longbench.filter(lambda x: x['length'] < 1024)

   model.config.output_attention = True
   directory = f"attention/{model_ckt.split('/')[-1]}"
   if not os.path.exists(directory):
      os.makedirs(directory)

   template = model2prompt[dataset]
   
   for sample in tqdm(longbench_filtered):
      prompt = template.format(**sample)
      prompt = build_chat(prompt)
      results, input_ids, all_layers_attentions, past_key_values = manual_infer_with_llama_with_attention(prompt)
      correlation_all = []
      x = torch.arange(0, len(all_layers_attentions)).cpu().detach().numpy()

      attention_sum = None 
      for layer_idx, attentions in enumerate(all_layers_attentions):
         attention = attentions 
         # breakpoint()
         attention_average = torch.mean(attention, dim=1)[0]
         size = torch.tril(torch.ones_like(attention_average))
         attention_average = (attention_average.sum(0) / size.sum(0)).cpu().detach().numpy()
         if attention_sum is None:
            attention_sum = attention_average
         else:
            attention_sum = attention_sum +  attention_average
      
      for sigma in tqdm([3.0, 4.0, 5.0, 6.0]):
         correlations_all_key = []
         correlations_all_value = []
         wandb.init(
            project='LLM-merge', 
            name=f'{model_ckt.split("/")[-1]}_{dataset}',
            reinit=True,
            config={
               'sigma': sigma,
               'model': model_ckt.split('/')[-1],
            }
         )
         # for sample in tqdm(longbench_filtered):
            # correlations_keys = []
            # correlations_values= []
         for layer_idx, attentions in enumerate(all_layers_attentions):
            attention = attentions 

            attention_average = torch.mean(attention, dim=1)

            attention_average = attention_average[0]

            attention = attention_average

            id2token = []
            for id in input_ids:
               id2token.append(tokenizer.decode(id.item()))

            id2token = id2token[0:]
            indices = list(range(len(id2token)))
            print(past_key_values[layer_idx][0].shape)

            energy_key = cal_energy(past_key_values[layer_idx][0], sigma=sigma).cpu().detach().numpy()
            energy_key_mean = cal_energy(past_key_values[layer_idx][0].mean(1), sigma=sigma).cpu().detach().numpy()
            energy_value = cal_energy(past_key_values[layer_idx][1], sigma=sigma).cpu().detach().numpy()

            size = torch.tril(torch.ones_like(attention))
            attention_layer_idx = (attention.sum(0) / size.sum(0)).cpu().detach().numpy()

            data = {
               'attention': attention_sum,
               'attention layer': attention_layer_idx,
               'key energy': energy_key.mean(1).squeeze(),
               'key mean energy': energy_key_mean.squeeze(),
               'value energy': energy_value.mean(1).squeeze(),
            } 
            # breakpoint()
            df = pd.DataFrame(data)
            key_correlation = df['attention'].corr(df['key energy'])
            value_correlation = df['attention'].corr(df['value energy'])
            key_mean_correlation = df['attention'].corr(df['key mean energy'])
            attention_correlation= df['attention'].corr(df['attention layer'])
            print(key_correlation, value_correlation, key_mean_correlation, attention_correlation)
            wandb.log({
               'layer idx':layer_idx,
               'key correlation': key_correlation, 
               'value correlation': value_correlation,
               # 'key_mean_correlation': key_mean_correlation,
               'attention layer': attention_correlation,
            })


      
         
   

