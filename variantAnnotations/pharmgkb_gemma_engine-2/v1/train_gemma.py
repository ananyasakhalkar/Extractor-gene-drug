
"""
Fine-tune Gemma 2 2B with LoRA for PharmGKB structured extraction.

Install:
  pip install -U transformers datasets peft accelerate bitsandbytes torch

You may need to accept the Gemma license on Hugging Face and authenticate:
  huggingface-cli login

Example:
  python train_gemma.py \
    --model google/gemma-2-2b \
    --train gemma_train.jsonl \
    --valid gemma_validation.jsonl \
    --out ./gemma-pharmgkb-lora
"""
import argparse, json, os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

def make_prompt(text):
    return (
        "You are a pharmacogenomics information extraction model. "
        "Read the PharmGKB sentence and extract the gene-drug relationship. "
        "Return ONLY valid JSON in this schema: "
        '{"interactions":[{"gene":"","drug":"","interaction_type":"","association":"",'
        '"direction":"","effect":"","variant_or_haplotype":"","phenotype":"","evidence_significance":""}]}'
        "\n\nSENTENCE:\n" + text.strip() + "\n\nJSON:\n"
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", default="./gemma-pharmgkb-lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=1024)
    args=ap.parse_args()

    data=load_dataset("json", data_files={
        "train":args.train, "validation":args.valid
    })

    tokenizer=AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model=AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto"
    )

    lora=LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        task_type="CAUSAL_LM"
    )
    model=get_peft_model(model,lora)
    model.print_trainable_parameters()

    def tokenize(batch):
        inputs=[]
        for text, output in zip(batch["text"], batch["output"]):
            prompt=make_prompt(text)
            target=json.dumps(output,ensure_ascii=False)
            inputs.append(prompt+target+tokenizer.eos_token)
        return tokenizer(inputs,truncation=True,max_length=args.max_length)

    tokenized=data.map(tokenize,batched=True,remove_columns=data["train"].column_names)

    train_args=TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        logging_steps=25,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(dtype==torch.float16),
        bf16=(dtype==torch.bfloat16),
        report_to="none",
        save_total_limit=2
    )
    collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
    trainer=Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

if __name__=="__main__":
    main()
