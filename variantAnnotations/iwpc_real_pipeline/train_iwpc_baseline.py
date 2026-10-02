
"""
Real IWPC warfarin baseline.
Usage:
python train_iwpc_baseline.py --phenotype path/to/phenotype.csv

The phenotype file must contain subject_id plus a continuous therapeutic/stable
warfarin dose field. Adjust DOSE_CANDIDATES below to match the source file.
"""
import argparse, pandas as pd, numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DOSE_CANDIDATES=[
    "Therapeutic Dose of Warfarin","Therapeutic Dose","Dose","dose",
    "Therapeutic Dose (mg/week)","Therapeutic Dose (mg/day)"
]

def find_col(df,cands):
    norm={str(c).strip().lower():c for c in df.columns}
    for x in cands:
        if x.lower() in norm:return norm[x.lower()]
    for c in df.columns:
        s=str(c).lower()
        if "therapeutic" in s and "dose" in s:return c
    raise ValueError("Could not identify therapeutic dose column. Columns: "+str(df.columns.tolist()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phenotype",required=True)
    ap.add_argument("--genotype",default="iwpc_model_ready_genotypes.csv")
    args=ap.parse_args()

    ph=pd.read_csv(args.phenotype)
    ge=pd.read_csv(args.genotype)
    ph.columns=[str(c).strip() for c in ph.columns]
    ge.columns=[str(c).strip() for c in ge.columns]

    if "subject_id" not in ph.columns:
        raise ValueError("Phenotype file needs subject_id so it can be linked to genotype data.")

    dose_col=find_col(ph,DOSE_CANDIDATES)
    ph["target_dose"]=pd.to_numeric(ph[dose_col],errors="coerce")
    df=ph.merge(ge,on="subject_id",how="inner",validate="one_to_one")
    df=df[df.target_dose.notna()].copy()

    # Only use features available before the outcome.
    candidates=[
        "age","Age","weight","Weight","height","Height","sex","Sex",
        "race","Race","indication","Indication","amiodarone","Amiodarone",
        "rs1799853_alt_count","rs1057910_alt_count","rs9923231_alt_count",
        "rs9934438_alt_count","rs2108622_alt_count","rs12777823_alt_count"
    ]
    cols=[c for c in candidates if c in df.columns]
    if not cols: raise ValueError("No recognized clinical/genotype columns found.")

    X=df[cols].copy()
    y=df["target_dose"]
    groups=df["subject_id"]

    # Patient-level split: no patient can appear in both train and test.
    splitter=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42)
    tr,te=next(splitter.split(X,y,groups=groups))
    Xtr,Xte,ytr,yte=X.iloc[tr],X.iloc[te],y.iloc[tr],y.iloc[te]

    cats=[c for c in cols if X[c].dtype=="object"]
    nums=[c for c in cols if c not in cats]
    prep=ColumnTransformer([
      ("num",SimpleImputer(strategy="median"),nums),
      ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                       ("oh",OneHotEncoder(handle_unknown="ignore"))]),cats)
    ])
    model=XGBRegressor(
      n_estimators=600,max_depth=4,learning_rate=.03,
      subsample=.8,colsample_bytree=.8,objective="reg:squarederror",
      random_state=42
    )
    pipe=Pipeline([("prep",prep),("model",model)])
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte)

    mae=mean_absolute_error(yte,pred)
    rmse=mean_squared_error(yte,pred)**.5
    r2=r2_score(yte,pred)
    print({"n_linked":len(df),"n_test":len(te),"MAE":mae,"RMSE":rmse,"R2":r2})
    pd.DataFrame({"subject_id":df.iloc[te]["subject_id"].values,
                  "actual_dose":yte.values,"predicted_dose":pred}).to_csv("iwpc_baseline_predictions.csv",index=False)

if __name__=="__main__":main()
