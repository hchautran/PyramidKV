from transformers.cache_utils import  Cache, DynamicCache, SinkCache
from typing import Tuple, Optional, List, Dict, Any,Callable
import torch
import torch.nn.functional as F


class PiToMeCache(Cache):
   """
   A cache that as described in the [Attention Sinks paper](https://arxiv.org/abs/2309.17453). It allows the model to
   generate beyond the length of its context window, without losing fluency in the conversation. As it discards past
   tokens, the model will lose the ability to generate tokens that depend on the context that was discarded.

   It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
   `[batch_size, num_heads, seq_len, head_dim]`.

   Parameters:
      window_length (`int`):
         The length of the context window.
      num_sink_tokens (`int`):
         The number of sink tokens. See the original paper for more information.

   Example:

      ```python
      >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache

      >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
      >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

      >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

      >>> # Prepare a cache class and pass it to model's forward
      >>> past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
      >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
      >>> outputs.past_key_values # access cache filled with key/values from generation
      SinkCache()
      ```
   """

   def __init__(self, window_length: int, num_sink_tokens: int, use_merge=False) -> None:
      super().__init__()
      self.key_cache: List[torch.Tensor] = []
      self.value_cache: List[torch.Tensor] = []
      self.window_length = window_length
      self.num_sink_tokens = num_sink_tokens
      self.cos_sin_rerotation_cache = {}
      self._cos_cache = None
      self._sin_cache = None
      self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
      self.energy_scores = None 
      self.indices = None
      self.local_size =  128 
      self.prune_size = window_length - self.local_size
      self.sizes =  None  
      self.use_merge = use_merge 

   @staticmethod
   def _rotate_half(x):
      x1 = x[..., : x.shape[-1] // 2]
      x2 = x[..., x.shape[-1] // 2 :]
      return torch.cat((-x2, x1), dim=-1)

   def _apply_key_rotary_pos_emb(
      self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
   ) -> torch.Tensor:
      rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
      return rotated_key_states

   def _get_rerotation_cos_sin(
      self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
   ) -> Tuple[torch.Tensor, torch.Tensor]:
      if key_states.shape[-2] not in self.cos_sin_rerotation_cache:
         # Upcast to float32 temporarily for better accuracy
         cos = cos.to(torch.float32)
         sin = sin.to(torch.float32)

         # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
         original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
         shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
         original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
         shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
         rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
         rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

         self.cos_sin_rerotation_cache[key_states.shape[-2]] = (
               rerotation_cos.to(key_states.dtype).unsqueeze(0),
               rerotation_sin.to(key_states.dtype).unsqueeze(0),
         )
      return self.cos_sin_rerotation_cache[key_states.shape[-2]]

   def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
      """Returns the sequence length of the cached states. A layer index can be optionally passed."""
      # TODO: deecate this function in favor of `cache_position`
      # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
      if len(self.key_cache) <= layer_idx:
         return 0
      return self.key_cache[layer_idx].shape[-2]
   
   def _cal_energy(self, metric:torch.Tensor, sigma:float=0.2):
      metric = F.normalize(metric, p=2, dim=-1) 
      sim = metric@metric.transpose(-1,-2)
      energy_score = (torch.exp(-(((1 - sim)/sigma)**2 * 0.5))).mean(-1) *  1/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))

      return energy_score,  sim

   def prune(self, x:torch.Tensor, indices:torch.Tensor ):
      mask = torch.ones(x.shape[-2], dtype=torch.bool)
      mask[indices] = False
      x = x[:, : , mask]
      return x
   
   def merge(self, x:torch.Tensor, x_src:torch.Tensor, x_dst:torch.Tensor, dst_idx:torch.Tensor, mode:str='mean'):
      B, T, C = x.shape
      x_dst = x_dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, indices.shape[-1], C), x_src, reduce=mode)
      return x

      
   def merge_wavg(self, x:torch.Tensor, dst_idx:torch.Tensor, size:torch.Tensor=None):
      if size is None:
         size = torch.ones_like(x[..., 0, None])
      x = self.merge(x*size, dst_idx, mode='sum')
      size = self.merge(size, dst_idx, mode='sum')
      x = x / size
      return x, size
   
   def get_max_length(self) -> Optional[int]:
      """Returns the maximum sequence length of the cached states."""
      return self.window_length

   def update(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      layer_idx: int,
      cache_kwargs: Optional[Dict[str, Any]] = None,
   ) -> Tuple[torch.Tensor, torch.Tensor]:
      """
      Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

      Parameters:
         key_states (`torch.Tensor`):
               The new key states to cache.
         value_states (`torch.Tensor`):
               The new value states to cache.
         layer_idx (`int`):
               The index of the layer to cache the states for.
         cache_kwargs (`Dict[str, Any]`, `optional`):
               Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
               `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
               rotation as the tokens are shifted.

      Return:
         A tuple containing the updated key and value states.
      """
      # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
      # with partially rotated position embeddings, like Phi or Persimmon.
      sin = cache_kwargs.get("sin")
      cos = cache_kwargs.get("cos")
      partial_rotation_size = cache_kwargs.get("partial_rotation_size")
      using_rope = cos is not None and sin is not None

      # Update the number of seen tokens
      if layer_idx == 0:
         self._seen_tokens += key_states.shape[-2]

      # Update the sin/cos cache, which holds sin/cos values for all possible positions
      if using_rope and layer_idx == 0:
         # BC: some models still pass `sin`/`cos` with 2 dims. In those models, they are the full sin/cos. Remove
         # after all RoPE models have a llama-like cache utilization.
         if cos.dim() == 2:
            self._cos_cache = cos
            self._sin_cache = sin
         else:
            if self._cos_cache is None:
               self._cos_cache = cos[0, ...]
               self._sin_cache = sin[0, ...]
            elif self._cos_cache.shape[0] < self.window_length:
               self._cos_cache = torch.cat([self._cos_cache, cos[0, ...]], dim=0)
               self._sin_cache = torch.cat([self._sin_cache, sin[0, ...]], dim=0)

      # [bsz, num_heads, seq_len, head_dim]
      if len(self.key_cache) <= layer_idx:
         # Empty cache
         self.key_cache.append(key_states)
         self.value_cache.append(value_states)
         self.sizes.append(None)

      elif key_states.shape[-2] + self.get_seq_length(layer_idx) < self.window_length:
         # Growing cache
         self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
         self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

      else:
         sink_keys = self.key_cache[layer_idx][:, :, : self.num_sink_tokens]
         sink_values = self.value_cache[layer_idx][:, :, : self.num_sink_tokens]

         if self.indices is not None: 
            keys = self.key_cache[layer_idx][
               :, :, self.num_sink_tokens:
            ]

            values = self.value_cache[layer_idx][
               :, :, self.num_sink_tokens:
            ]

            local_keys= keys[:, :, -self.local_size:]
            local_values = values[:, :, -self.local_size:]
            remain_keys= keys[:, :, :-self.local_size]
            remain_values = values[:, :, :-self.local_size:]


            if not self.use_merge:
               remain_keys = self.prune(remain_keys, indices=self.indices)
               remain_values = self.prune(remain_values, indices=self.indices) 
            else:
               remain_keys = self.merge(remain_keys, indices=self.indices)
               remain_values = self.merge(remain_values, indices=self.indices) 

            keys_to_keep = torch.cat((remain_keys, local_keys), dim=-2)
            values_to_keep = torch.cat((remain_values, local_values), dim=-2)

            energy, _ = self._cal_energy(remain_keys.mean(1), sigma=0.1)
            if self.energy_scores is None:
               self.energy_scores = energy 
            else:
               self.energy_scores = self.energy_scores + energy
            
            if layer_idx == len(self.key_cache) - 1: 
               self.indices = torch.topk(self.energy_scores.mean(-1), k=key_states.shape[-2], largest=True).indices 

         else:
            keys_to_keep = self.key_cache[layer_idx][
               :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2] :
            ]
            values_to_keep = self.value_cache[layer_idx][
               :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2] :
            ]

            if layer_idx == len(self.key_cache) - 1: 
               self.indices = 0 



         # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
         if using_rope:
               rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
                  key_states, self._cos_cache[: self.window_length+2], self._sin_cache[: self.window_length+2]
               )
               if partial_rotation_size is not None:
                  keys_to_keep, keys_pass = (
                     keys_to_keep[..., :partial_rotation_size],
                     keys_to_keep[..., partial_rotation_size:],
                  )
               keys_to_keep = self._apply_key_rotary_pos_emb(keys_to_keep, rerotation_cos, rerotation_sin)

               if partial_rotation_size is not None:
                  keys_to_keep = torch.cat((keys_to_keep, keys_pass), dim=-1)


         # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
         self.key_cache[layer_idx] = torch.cat([sink_keys, keys_to_keep, key_states], dim=-2)
         self.value_cache[layer_idx] = torch.cat([sink_values, values_to_keep, value_states], dim=-2)

      return self.key_cache[layer_idx], self.value_cache[layer_idx]
   
   