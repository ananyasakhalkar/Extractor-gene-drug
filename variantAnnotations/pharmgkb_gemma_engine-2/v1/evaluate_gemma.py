
import argparse,json,re
from collections import defaultdict

FIELDS=["gene","drug","interaction_type","association","direction"]

def norm(x):
    return re.sub(r"\s+"," ",str(x or "").strip().lower())

def flatten(ex):
    out=set()
    for i in ex.get("interactions",[]):
        # drug can be list or string
        drugs=i.get("drug",[])
        if not isinstance(drugs,list): drugs=[drugs]
        for d in drugs:
            out.add(tuple(norm(i.get(f,"")) for f in FIELDS[:-3]) + (norm(d),) +
                    tuple(norm(i.get(f,"")) for f in FIELDS[2:]))
    return out

def field_metrics(gold,pred,field):
    g={norm(x.get(field,"")) for x in gold.get("interactions",[]) if norm(x.get(field,""))}
    p={norm(x.get(field,"")) for x in pred.get("interactions",[]) if norm(x.get(field,""))}
    tp=len(g&p); fp=len(p-g); fn=len(g-p)
    pr=tp/(tp+fp) if tp+fp else 0
    rc=tp/(tp+fn) if tp+fn else 0
    f1=2*pr*rc/(pr+rc) if pr+rc else 0
    return pr,rc,f1,tp,fp,fn

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold",required=True)
    ap.add_argument("--pred",required=True)
    args=ap.parse_args()

    gold=[json.loads(x) for x in open(args.gold,encoding="utf-8") if x.strip()]
    pred=[json.loads(x) for x in open(args.pred,encoding="utf-8") if x.strip()]
    assert len(gold)==len(pred)

    for field in ["gene","drug","interaction_type","association","direction"]:
        vals=[field_metrics(g,p,field) for g,p in zip(gold,pred)]
        pr=sum(v[0] for v in vals)/len(vals)
        rc=sum(v[1] for v in vals)/len(vals)
        f1=sum(v[2] for v in vals)/len(vals)
        print(f"{field:20s} precision={pr:.4f} recall={rc:.4f} f1={f1:.4f}")

    tp=fp=fn=0
    for g,p in zip(gold,pred):
        gs=flatten(g); ps=flatten(p)
        tp+=len(gs&ps); fp+=len(ps-gs); fn+=len(gs-ps)
    pr=tp/(tp+fp) if tp+fp else 0
    rc=tp/(tp+fn) if tp+fn else 0
    f1=2*pr*rc/(pr+rc) if pr+rc else 0
    print(f"\nRELATION (strict fields) precision={pr:.4f} recall={rc:.4f} f1={f1:.4f}")
    print(f"TP={tp} FP={fp} FN={fn}")

if __name__=="__main__":
    main()
