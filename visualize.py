
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.llama.pitomekv import convert
from accelerate import Accelerator
import os

accelerator = Accelerator()
model_checkpoint_path = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path, torch_dtype=torch.float16)
accelerator.prepare(model)
convert(model)
model.config.output_attention = True

directory = "attention"


if not os.path.exists(directory):
   os.makedirs(directory)



def manual_infer_with_llama_with_attention(prompt, max_length=50):

   input_ids = tokenizer.encode(prompt, return_tensors='pt').to(accelerator.device)
   all_layers_attentions = [] 

   for _ in range(max_length):

      raw_outputs = model(input_ids, output_attentions=True, return_dict=True)
      print(raw_outputs)
      output = raw_outputs.logits
      next_token_logits = output[:, -1, :]
      
      attentions = raw_outputs.attentions
   

      next_token = torch.argmax(next_token_logits, dim=-1)

      input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

      if next_token in tokenizer.all_special_ids:
         break
      
   for i in range(len(attentions)):
      all_layers_attentions.append(attentions[i].detach().cpu())
   return tokenizer.decode(input_ids[0], skip_special_tokens=True), input_ids[0], all_layers_attentions

# 3document in example

   

input_prompt = """
[INST] <<SYS>>
             You are given some documents, and you need to answer a question based on these documents.
            Your answer should be less than five words.
              
<</SYS>>
Document: Roman Republic After having declined in size following the subjugation of the Mediterranean, the Roman navy underwent short-term upgrading and revitalisation in the late Republic to meet several 
new demands. Under Caesar, an invasion fleet was assembled in the English Channel to allow the invasion of Britannia; under Pompey, a large fleet was raised in the Mediterranean Sea to clear the sea of Cili
cian pirates. During the civil war that followed, as many as a thousand ships were either constructed or pressed into service from Greek cities. 
Document: North Sea The North Sea is bounded by the Orkney Islands and east coast of Great Britain to the west and the northern and central European mainland to the east and south, including Norway, Denmark
, Germany, the Netherlands, Belgium, and France. In the southwest, beyond the Straits of Dover, the North Sea becomes the English Channel connecting to the Atlantic Ocean. In the east, it connects to the Ba
ltic Sea via the Skagerrak and Kattegat, narrow straits that separate Denmark from Norway and Sweden respectively. In the north it is bordered by the Shetland Islands, and connects with the Norwegian Sea, w
hich lies in the very north - eastern part of the Atlantic. 
Question: Who sent naval ships to the body of water that joins the Atlantic and the sea where the Rhine ends? 
Answer:  [/INST]
"""

results, input_ids, all_layers_attentions = manual_infer_with_llama_with_attention(input_prompt)

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

   import matplotlib.pyplot as plt
   import seaborn as sns
   import numpy as np


   num_heads = 1
   sequence_length = 10

   attention = attention.cpu().detach().numpy()

   plt.figure(figsize=(100, 80))

   fig, ax = plt.subplots()
   ax.imshow(attention, vmax=100)
   ax.set_xticks(np.arange(len(id2token)), labels=[])
   ax.set_yticks(np.arange(len(id2token)), labels=[])



   plt.title(f'Attention Weights Heatmap Layer {layer_idx}')
   plt.xlabel('Key Positions')
   plt.ylabel('Query Positions')
   plt.savefig(f'attention/layer{layer_idx}.png', dpi=300, format='png')
