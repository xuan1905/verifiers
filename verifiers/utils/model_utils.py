from importlib.util import find_spec
from typing import Dict, Any, Union, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def is_liger_available() -> bool:
    return find_spec("liger_kernel") is not None

def get_model(model_name: str, model_kwargs: Union[Dict[str, Any], None] = None) -> Any:
    if model_kwargs is None:
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
    if is_liger_available():
        print("Using Liger kernel")
        from liger_kernel.transformers import AutoLigerModelForCausalLM # type: ignore
        return AutoLigerModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
def get_tokenizer(model_name: str) -> Any:
    if "Instruct" in model_name:
        return AutoTokenizer.from_pretrained(model_name)
    else:
        try:
            return AutoTokenizer.from_pretrained(model_name + "-Instruct")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # check if tokenizer has chat_template attribute
            if hasattr(tokenizer, "chat_template"):
                return tokenizer
            else:
                raise ValueError(f"Tokenizer for model {model_name} does not have chat_template attribute, \
                                  and could not find a tokenizer with the same name as the model with suffix \
                                 '-Instruct'. Please provide a tokenizer with the chat_template attribute.")
            
def get_model_and_tokenizer(model_name: str, model_kwargs: Union[Dict[str, Any], None] = None) -> Tuple[Any, Any]:
    model = get_model(model_name, model_kwargs)
    tokenizer = get_tokenizer(model_name)
    return model, tokenizer