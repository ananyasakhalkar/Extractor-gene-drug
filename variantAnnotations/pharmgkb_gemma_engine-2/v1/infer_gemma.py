
import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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
    ap.add_argument("--adapter",required=True)
    ap.add_argument("--base-model",default="google/gemma-2-2b")
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--max-new-tokens",type=int,default=256)
    args=ap.parse_args()

    tok=AutoTokenizer.from_pretrained(args.adapter)
    base=AutoModelForCausalLM.from_pretrained(
        args.base_model,torch_dtype="auto",device_map="auto"
    )
    model=PeftModel.from_pretrained(base,args.adapter)
    model.eval()

    with open(args.input,encoding="utf-8") as fin, open(args.output,"w",encoding="utf-8") as fout:
        for line in fin:
            ex=json.loads(line)
            prompt=make_prompt(ex["text"])
            inputs=tok(prompt,return_tensors="pt").to(model.device)
            with torch.no_grad():
                out=model.generate(**inputs,max_new_tokens=args.max_new_tokens,
                                   do_sample=False,temperature=0.0)
            text=tok.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
            try:
                pred=json.loads(text)
            except Exception:
                pred={"parse_error":True,"raw_output":text}
            fout.write(json.dumps(pred,ensure_ascii=False)+"\n")

if __name__=="__main__":
    main()
