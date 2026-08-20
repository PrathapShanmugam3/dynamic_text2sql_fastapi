import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from app import config
from app import prompt as prompt_builder

BASE_MODEL = config.BASE_MODEL
ADAPTER_PATH = config.ADAPTER_PATH


def _patch_bnb_frozenset_bug():
    """transformers==4.46.3's bnb integration code calls .discard() on the set
    of bitsandbytes-supported devices, but some bitsandbytes builds return
    that as a frozenset (immutable, no .discard()), crashing
    from_pretrained(..., quantization_config=...) with AttributeError:
    'frozenset' object has no attribute 'discard'. We're loading a single,
    valid BitsAndBytesConfig on one CUDA GPU, so this multi-backend
    availability check has nothing meaningful to reject here -- skip it if it
    hits the known frozenset bug. Idempotent: safe to call more than once
    (e.g. if SQLGenerator is constructed twice in the same process) since a
    second call is a no-op once the marker attribute is set."""
    import transformers.integrations.bitsandbytes as bnb_integration

    if getattr(
        bnb_integration._validate_bnb_multi_backend_availability,
        "_is_frozenset_shim",
        False,
    ):
        return

    original_validate = bnb_integration._validate_bnb_multi_backend_availability

    def patched_validate(raise_exception=True, *args, **kwargs):
        try:
            return original_validate(raise_exception, *args, **kwargs)
        except AttributeError as exc:
            if "discard" not in str(exc):
                raise
            print(f"Skipping bnb multi-backend availability check (known bug: {exc})")
            return True

    patched_validate._is_frozenset_shim = True
    bnb_integration._validate_bnb_multi_backend_availability = patched_validate


class SQLGenerator:
    def __init__(self):
        use_gpu = torch.cuda.is_available() and not config.FORCE_CPU

        if use_gpu:
            self._load_gpu_quantized()
        else:
            self._load_cpu()

        self.model.eval()

    def _load_gpu_quantized(self):
        """GPU path: 4-bit NF4 base model + PEFT LoRA adapter on top, matching
        the training-time quantization config exactly."""
        _patch_bnb_frozenset_bug()

        print(f"[GPU] Loading base model: {BASE_MODEL}")
        print(f"[GPU] Loading LoRA adapter: {ADAPTER_PATH}")

        if not os.path.exists(ADAPTER_PATH):
            raise FileNotFoundError(
                f"LoRA adapter not found: {ADAPTER_PATH}. "
                "Set ADAPTER_PATH to your checkpoint directory."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        self.model = PeftModel.from_pretrained(base, ADAPTER_PATH)

    def _load_cpu(self):
        """CPU path: bitsandbytes 4-bit quantization requires CUDA, so this
        loads the already-merged full-precision model instead of base+adapter.
        Set MERGED_MODEL_PATH (or MODEL_PATH) to a `save_pretrained` merged
        model directory. Expect generation to be far slower than on GPU."""
        model_path = config.MERGED_MODEL_PATH
        if not model_path:
            raise FileNotFoundError(
                "CPU inference requires a merged model. Set MERGED_MODEL_PATH "
                "(or MODEL_PATH) to a directory produced by merging the LoRA "
                "adapter into the base model and calling save_pretrained()."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Merged model not found: {model_path}")

        print(f"[CPU] Loading merged model: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

    def generate_from_prompt(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_sql(self, question: str, database_type: str, schema: dict) -> str:
        prompt = prompt_builder.build_prompt(question, database_type, schema)
        return self.generate_from_prompt(prompt)
