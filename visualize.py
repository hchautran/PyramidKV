
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.cache import PiToMeCache
from models.llama.pitomekv import convert
from accelerate import Accelerator
from const import  (
   LLAMA2_7B,
   LLAMA3_8B,
   LLAMA3_1_8B,
   LLAMA3_2_3B,
   LLAMA3_2_1B
)
import os

accelerator = Accelerator()
# model_ckt = LLAMA3_1_8B 
model_ckt = LLAMA2_7B 
tokenizer = AutoTokenizer.from_pretrained(model_ckt)
model = AutoModelForCausalLM.from_pretrained(model_ckt, torch_dtype=torch.float16)
accelerator.prepare(model)
convert(model)
model.config.output_attention = True

directory = f"attention/{model_ckt.split('/')[-1]}"



if not os.path.exists(directory):
   os.makedirs(directory)



def cal_energy(metric:torch.Tensor, sigma:float=0.1):
   metric = F.normalize(metric, p=2, dim=-1) 
   sim = metric@metric.transpose(-1,-2)
   energy_score = (torch.exp(-(((1 - sim)/sigma)**2 * 0.5))).mean(-1) *  1/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))
   return energy_score

   


def manual_infer_with_llama_with_attention(prompt, max_length=50):
   past_key_values = PiToMeCache()

   input_ids = tokenizer.encode(prompt, return_tensors='pt').to(accelerator.device)
   all_layers_attentions = [] 

   for _ in range(max_length):

      raw_outputs = model(input_ids, output_attentions=True, return_dict=True, use_cache=True, past_key_values=past_key_values)
      output = raw_outputs.logits
      next_token_logits = output[:, -1, :]
      
      attentions = raw_outputs.attentions

      next_token = torch.argmax(next_token_logits, dim=-1)

      input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

      if next_token in tokenizer.all_special_ids:
         break
      
   for i in range(len(attentions)):
      all_layers_attentions.append(attentions[i].detach().cpu())
   
   accelerator.clear()
   return tokenizer.decode(input_ids[0], skip_special_tokens=True), input_ids[0], all_layers_attentions, raw_outputs.past_key_values

   
input_prompt = """
[INST] <<SYS>>
             You are given some documents, and you need to answer a question based on these documents.
            Your answer should be less than five words.
              
<</SYS>>
Document: Roman Republic After having declined in size following the subjugation of the Mediterranean, the Roman navy underwent short-term upgrading and revitalisation in the late Republic to meet several 
new demands. Under Caesar, an invasion fleet was assembled in the English Channel to allow the invasion of Britannia; under Pompey, a large fleet was raised in the Mediterranean Sea to clear the sea of Cili
cian pirates. During the civil war that followed, as many as a thousand ships were either constructed or pressed into service from Greek cities. 
Question: Who sent naval ships to the body of water that joins the Atlantic and the sea where the Rhine ends? 
Answer:  [/INST]
"""



results, input_ids, all_layers_attentions, past_key_values = manual_infer_with_llama_with_attention(input_prompt)



import numpy as np 
for layer_idx, attentions in enumerate(all_layers_attentions):
   attention = attentions * 10000

   attention_average = torch.mean(attention, dim=1)

   attention_average = attention_average[0]

   attention = attention_average

   id2token = []
   for id in input_ids:
      id2token.append(tokenizer.decode(id.item()))

   id2token = id2token[0:]
   indices = list(range(len(id2token)))



   num_heads = 1
   sequence_length = 10

   # attention = attention.cpu().detach().numpy()a
   energy_key = cal_energy(past_key_values[layer_idx][0], sigma=0.1).cpu().detach().numpy()
   energy_value = cal_energy(past_key_values[layer_idx][1], sigma=0.1).cpu().detach().numpy()

   fig = plt.figure(figsize=(9, 10))
   gs = gridspec.GridSpec(9, 10)

   ax1 = fig.add_subplot(gs[:-1, :])
   ax2 = fig.add_subplot(gs[-1, :], sharex=ax1)
   # ax3 = fig.add_subplot(gs[-1, :], sharex=ax1)

   ax1.imshow(attention, vmax=100)
   ax2.imshow(energy_key.reshape(1, -1), cmap='inferno', aspect='auto')
   # ax3.imshow(energy_value.reshape(1, -1), cmap='inferno', aspect='auto')
   plt.setp(ax1.get_xticklabels(), visible=False)
   

   plt.tight_layout()
   plt.savefig(f'{directory}/layer{layer_idx}.png', dpi=300, format='png')
   plt.show()