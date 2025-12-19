from transformers import AutoTokenizer, AutoModelForCausalLM
#from vllm import LLM, SamplingParams
import torch
from typing import List, Dict, Any, Tuple

class LlamaInference:
    def __init__(
        self,
        model_id: str = "",
        device_map: str = "auto",
        batch_inference: bool = False,
        use_vllm: bool = False,
        max_model_len: int = 4096
    ):
        self.model_id = model_id
        self.device_map = device_map
        self.batch_inference = batch_inference
        self.use_vllm = use_vllm
        self.max_model_len = max_model_len
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if self.use_vllm:
            model = LLM(model=self.model_id, gpu_memory_utilization=0.9, max_model_len=self.max_model_len, tensor_parallel_size=torch.cuda.device_count())
            tokenizer = AutoTokenizer.from_pretrained(self.model_id , cache_dir='/home/jovyan/new-model-weights')
        elif self.batch_inference:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left" , cache_dir='/home/jovyan/new-model-weights')
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id , cache_dir='/home/jovyan/new-model-weights')
            
        if not self.use_vllm:    
            model = AutoModelForCausalLM.from_pretrained(self.model_id,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,  cache_dir='/home/jovyan/new-model-weights')

        return model, tokenizer


    def generate(
        self,
        messages: List[Dict[str, Any]],
        num_return_sequences: int = 1,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0,
        top_p: float = 0.9,
        seed: int = 42,
    ) -> str:
        if self.use_vllm:
            sampling_params = SamplingParams(
                temperature=temperature,  
                max_tokens=max_new_tokens,
                n = num_return_sequences,
                seed = seed,
                stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            )

            conversations = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            # Remove duplicated start token for meta-llama/llama-3.1-8b-instrct vllm
            if self.model_id == "meta-llama/llama-3.1-8b-instrct":
                conversations = self.tokenizer(conversations)
                conversations = conversations['input_ids'][1:]
                output = self.model.generate(prompt_token_ids=conversations, sampling_params=sampling_params)
            else:
                output = self.model.generate([conversations], sampling_params)

            return output[0].outputs[0].text

        else:        
            input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
            ).to(self.model.device)
        
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=False,
                top_p=top_p,
                use_cache=True,
            )
            
            response = outputs[0][input_ids.shape[-1]:]
            output = self.tokenizer.decode(response, skip_special_tokens=True)
        
            return output
        

    def batch_inference(
        self,
        messages: List[List[Dict[str, Any]]],
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float = 0.4,
        top_p: float = 0.9,    
    ) -> List[str]:
        if self.use_vllm:
            sampling_params = SamplingParams(
                temperature=temperature,  
                max_tokens=max_new_tokens,
                top_p=top_p,
                stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            )

            conversations = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            if self.model_id == "meta-llama/llama-3.1-8b-instrct":
                conversations = [self.tokenizer(conversation) for conversation in conversations]
                conversations = [conversation['input_ids'][1:] for conversation in conversations]
                outputs = self.model.generate(prompt_token_ids=conversations, sampling_params=sampling_params)
            else:
                outputs = self.model.generate(conversations, sampling_params)

            gen_text = []
            for output in outputs:
                gen_text.append(output.outputs[0].text)

            return gen_text
        
        else:
            texts = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            inputs = self.tokenizer(texts, padding="longest", return_tensors="pt")
            inputs = {key: val.cuda() for key, val in inputs.items()}
            temp_texts=self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id, 
                eos_token_id=terminators,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
            )
            
            gen_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            gen_text = [i[len(temp_texts[idx]):] for idx, i in enumerate(gen_text)]
            
            return gen_text
