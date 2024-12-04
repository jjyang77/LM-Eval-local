import torch
from vllm import LLM, SamplingParams
model_name = "Qwen/Qwen2.5-7B-Instruct"
max_length = 8192

model = LLM(model=model_name, max_model_len=max_length)
tokenizer = model.get_tokenizer()

messages_list =[[{"role": "user", "content": "Food and Entertaining: How to make french toast sticks. Prepare your pan or griddle. Set the heat to medium high, and coat the surface with cooking spray. Cut each slice of bread into thirds and set aside. \nChoose the best continuation:\n A. Arrange your french toast sticks in a single layer on a large plate or cutting board. If you don't have a cutting board, use a knife to distribute the rolls in an even layer in the pan.\n B. Drain more flaked butter and mix in the meringue mix. Place the butter evenly over the bread slices and cover the pan.\n C. Place a dry seasoning blend or breadcrumbs on the wet side of a bread roll. Sprinkle the seasonings on both sides of the bread roll.\n D. Try to use a thicker toast slices; they will hold up a lot better. Many people also find that stale, day-old toast yields crisper and sturdier french toast sticks.\nAnswer:"}]]
formatted_prompts = [tokenizer.apply_chat_template(p, tokenize=False,  add_generation_prompt=True) for p in messages_list]

sampling_params = SamplingParams(max_tokens=1,   temperature=0.0,  logprobs=20)
outputs = model.generate(prompts=formatted_prompts, sampling_params=sampling_params, use_tqdm=False)
print(outputs[0].outputs[0].logprobs)
#[{785: Logprob(logprob=-0.0071074506267905235, rank=1, decoded_token='The'), 16141: Logprob(logprob=-5.132107257843018, rank=2, decoded_token='Answer'), 35: Logprob(logprob=-7.382107257843018, rank=3, decoded_token='D'), 32: Logprob(logprob=-7.507107257843018, rank=4, decoded_token='A'), ...]
sampling_params = SamplingParams(max_tokens=1,   temperature=0.0,  logprobs=10, logit_bias={enc: 5 for enc in [32,33,34,35]})
outputs = model.generate(prompts=formatted_prompts, sampling_params=sampling_params, use_tqdm=False)
print(outputs[0].outputs[0].logprobs)
#[{785: Logprob(logprob=-0.15711037814617157, rank=1, decoded_token='The'), 35: Logprob(logprob=-2.6571104526519775, rank=2, decoded_token='D'), 32: Logprob(logprob=-2.6571104526519775, rank=3, decoded_token='A'),...]


del model, tokenizer, outputs
torch.cuda.empty_cache()

from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_ids = tokenizer(formatted_prompts[0], return_tensors="pt").input_ids
outputs = model.generate(
    input_ids, 
    max_new_tokens=1, 
    return_dict_in_generate=True, 
    output_logits=True,    
)

# Access logits for each step
logprobs = torch.nn.functional.log_softmax(outputs.logits[0], dim=1)
print(logprobs[0][32])
#tensor(-7.2413)
print(logprobs[0][35])
#tensor(-7.2699)

for enc in [32,33,34,35]:
    outputs.logits[0][0][enc] = outputs.logits[0][0][enc] + 5

logprobs2 = torch.nn.functional.log_softmax(outputs.logits[0], dim=1)
print(logprobs2[0][32])
#tensor(-2.4305)
print(logprobs2[0][35])
#tensor(-2.4591)
