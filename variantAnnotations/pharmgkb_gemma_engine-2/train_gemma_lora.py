
import json, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def format_example(ex):
    prompt = (
        "Extract all gene-drug interactions from the passage. "
        "Return ONLY valid JSON with an 'interactions' list. "
        "Each item must contain gene, drug, interaction_type, direction, "
        "effect, and evidence_level.\n\nPASSAGE:\n" + ex["text"] +
        "\n\nJSON:\n"
    )
    target = json.dumps(ex["output"], ensure_ascii=False)
    return {"prompt": prompt, "target": target}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2b")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./gemma2b-pharmgkb-lora")
    ap.add_argument("--epochs", type=int, default=3)
    args=ap.parse_args()

    ds=load_dataset("json", data_files=args.data)["train"]
    ds=ds.map(format_example)

    tokenizer=AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    model=AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )

    lora=LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        task_type="CAUSAL_LM"
    )
    model=get_peft_model(model,lora)
    model.print_trainable_parameters()

    def tokenize(batch):
        texts=[p+t+tokenizer.eos_token for p,t in zip(batch["prompt"],batch["target"])]
        return tokenizer(texts, truncation=True, max_length=2048)

    tokenized=ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    args_train=TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=20,
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )
    trainer=Trainer(model=model,args=args_train,train_dataset=tokenized)
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

if __name__=="__main__":
    main()
