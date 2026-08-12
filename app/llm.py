import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_PATH = os.getenv(
    "ADAPTER_PATH",
    "/models/texttosql/results/checkpoint-3480"
)

class SQLGenerator:
    def __init__(self):
        print(f"Loading base model: {BASE_MODEL}")
        print(f"Loading LoRA adapter: {ADAPTER_PATH}")

        if not os.path.exists(ADAPTER_PATH):
            raise FileNotFoundError(
                f"LoRA adapter not found: {ADAPTER_PATH}. "
                "Set ADAPTER_PATH to your checkpoint-3480 directory."
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
            device_map="auto"
        )

        self.model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        self.model.eval()

    @staticmethod
    def schema_to_text(schema: dict) -> str:
        parts = []
        for table, columns in schema.items():
            parts.append(f"TABLE {table}")
            for col in columns:
                parts.append(f"  - {col['name']} {col['type']}")
            parts.append("")
        return "\n".join(parts)

    def generate_sql(self, question: str, database_type: str, schema: dict) -> str:
        schema_text = self.schema_to_text(schema)

        prompt = f"""You are a professional Text-to-SQL system.

Database type:
{database_type}

Database schema:
{schema_text}

Rules:
1. Generate only valid {database_type} SQL.
2. Use only tables from the provided schema.
3. Use only columns from the provided schema.
4. Never invent table names.
5. Never invent column names.
6. Generate SELECT queries only.
7. Do not generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE.
8. Return SQL only. No explanation.

User question:
{question}

SQL:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return generated
