
import json, argparse
from collections import Counter

def canon(x):
    return str(x).strip().lower()

def key(x):
    return (canon(x.get("gene","")), canon(x.get("drug","")),
            canon(x.get("interaction_type","")))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args=ap.parse_args()

    gold=[json.loads(x) for x in open(args.gold) if x.strip()]
    pred=[json.loads(x) for x in open(args.pred) if x.strip()]

    tp=fp=fn=0
    for g,p in zip(gold,pred):
        gs=set(key(x) for x in g.get("interactions",[]))
        ps=set(key(x) for x in p.get("interactions",[]))
        tp += len(gs & ps)
        fp += len(ps-gs)
        fn += len(gs-ps)

    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    print({"precision":precision,"recall":recall,"f1":f1,
           "tp":tp,"fp":fp,"fn":fn})

if __name__=="__main__":
    main()
