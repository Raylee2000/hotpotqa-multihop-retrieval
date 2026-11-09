"""Predeclared question-paired bootstrap, exact paired binary tests, reports."""
from __future__ import annotations
from math import isfinite
import numpy as np
from scipy.stats import binomtest

B,SEED=2000,42


def bootstrap(values):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("bootstrap requires finite paired observations")
    rng=np.random.default_rng(SEED)
    dist=np.concatenate([x[rng.integers(0,len(x),size=(min(64,B-i),len(x)))].mean(1) for i in range(0,B,64)])
    return {"n":len(x),"sum":float(x.sum()),"mean":float(x.mean()),
            "ci95":[float(v) for v in np.percentile(dist,[2.5,97.5])]}


def paired(full,other):
    a,b=np.asarray(full,dtype=int),np.asarray(other,dtype=int)
    assert a.shape==b.shape and len(a)>0
    fix=int(((a==1)&(b==0)).sum())
    regression=int(((a==0)&(b==1)).sum())
    return {**bootstrap(a-b),"full_fixes":fix,"full_regressions":regression,
            "mcnemar_exact_p":float(binomtest(fix,fix+regression,0.5).pvalue) if fix+regression else 1.0}


def summarize(rows,full="R5_full"):
    ids=[r["qid"] for r in rows]
    assert len(ids)==len(set(ids)) and len(ids)>0
    methods=list(rows[0]["methods"])
    assert all(set(r["methods"])==set(methods) for r in rows)
    out={"n_questions":len(rows),"statistics":{"bootstrap_resamples":B,"seed":SEED,"unit":"question", "paired_test":"exact McNemar", "primary":"R5_full - R0 (Matrix A)","other_comparisons":"secondary/exploratory; Holm family per stratum across all full-vs-other comparisons"},"strata":{}}
    for group in ("overall","bridge","comparison"):
        sub=rows if group=="overall" else [r for r in rows if r["qtype"]==group]
        if not sub:
            continue
        r={"n":len(sub),"methods":{},"full_vs":{},"component_comparisons":{}}
        for method in methods:
            vals=[q["methods"][method] for q in sub]
            e={"pr1":bootstrap([v["pr1"] for v in vals])}
            for k in ("pr3","pr5","rr","cand"):
                present=[float(v[k]) for v in vals if k in v]
                if present:
                    e[k]={"mean":float(np.mean(present)),"n":len(present)}
            cm=sum(not v.get("cand",True) for v in vals)
            rm=sum(bool(v.get("cand",True)) and not v["pr1"] for v in vals)
            e["failure_decomposition"]={"candidate_misses":cm,"ranking_misses":rm,"correct":sum(v["pr1"] for v in vals),
                                        "one_hop_ranking_errors":sum(bool(v.get("one_hop_error")) for v in vals)}
            r["methods"][method]=e
            if method!=full:
                r["full_vs"][method]=paired([q["methods"][full]["pr1"] for q in sub],[q["methods"][method]["pr1"] for q in sub])
        p=r["full_vs"]
        running=0.
        for i,key in enumerate(sorted(p,key=lambda k:p[k]["mcnemar_exact_p"])):
            running=max(running,min(1.,(len(p)-i)*p[key]["mcnemar_exact_p"]))
            p[key]["holm_p"]=running
        comparisons={"encoder_ft":("r0/general|general|TOP2","r0/frozen|frozen|TOP2"),
                     "first_hop_ft":("r0/general|general|TOP2","r0/frozen|general|TOP2"),
                     "second_hop_ft":("r0/general|general|TOP2","r0/general|frozen|TOP2"),
                     "sf_vs_full_body":("r0/general|csh|TOP2","r0/general|csh|QV0"),
                     "sf_vs_question_only":("r0/general|csh|TOP2","r0/general|csh|QQ"),
                     "dynamic_vs_top2":("r0/general|csh|DYN","r0/general|csh|TOP2"),
                     "csh_vs_general":("r0/general|csh|TOP2","r0/general|general|TOP2")}
        for label,(a,b) in comparisons.items():
            if a in methods and b in methods:
                r["component_comparisons"][label]={"a":a,"b":b,**paired([q["methods"][a]["pr1"] for q in sub],[q["methods"][b]["pr1"] for q in sub])}
        out["strata"][group]=r
    if "bridge" in out["strata"] and "comparison" in out["strata"]:
        out["balanced_macro_pr1"]={m:(out["strata"]["bridge"]["methods"][m]["pr1"]["mean"]+out["strata"]["comparison"]["methods"][m]["pr1"]["mean"])/2 for m in methods}
    return out


def markdown(title,summary,provenance):
    lines=[f"# {title}","",provenance,"", "Statistics: question-paired bootstrap 2,000, seed 42. V-select is selection data; B/C has prior QType evaluation exposure. Candidate-set diagnostics do not isolate ranking causally.","",
           "| Method | Overall PR@1 [95% CI] | Correct / N | Bridge PR@1 | Comparison PR@1 | Candidate recall |", "|---|---|---|---|---|---|"]
    for m,e in summary["strata"]["overall"]["methods"].items():
        p=e["pr1"]
        bridge=summary["strata"].get("bridge",{}).get("methods",{}).get(m,{}).get("pr1",{}).get("mean")
        comp=summary["strata"].get("comparison",{}).get("methods",{}).get(m,{}).get("pr1",{}).get("mean")
        lines.append(f"| {m} | {p['mean']:.6f} [{p['ci95'][0]:.6f}, {p['ci95'][1]:.6f}] | {int(p['sum'])}/{p['n']} | {bridge} | {comp} | {e.get('cand',{}).get('mean')} |")
    lines += ["", "| Full vs method | Paired delta [95% CI] | Fix / regression | Holm p |", "|---|---|---|---|"]
    for m,p in summary["strata"]["overall"]["full_vs"].items():
        lines.append(f"| {m} | {p['mean']:.6f} [{p['ci95'][0]:.6f}, {p['ci95'][1]:.6f}] | {p['full_fixes']} / {p['full_regressions']} | {p['holm_p']:.6g} |")
    lines += ["", "Matched component comparisons (R0; candidate sets may change):", "", "| Comparison | Delta [95% CI] |", "|---|---|"]
    for m,p in summary["strata"]["overall"]["component_comparisons"].items():
        lines.append(f"| {m} | {p['mean']:.6f} [{p['ci95'][0]:.6f}, {p['ci95'][1]:.6f}] |")
    return "\n".join(lines)+"\n"


def conditional_summary(pipeline):
    output={}
    methods=sorted({m for r in pipeline for m in r["cond"] if m.split("|")[0] in ("frozen","general","csh") or m.endswith("|TOP2")})
    for group in ("overall","bridge","comparison"):
        rs=pipeline if group=="overall" else [r for r in pipeline if r["qtype"]==group]
        if not rs:
            continue
        result={}
        for name in methods:
            values=[[x["rank"] for x in r["cond"].get(name,[])] for r in rs]
            counts=np.array([len(v) for v in values])
            total=int(counts.sum())
            if not total:
                result[name]={"status":"NOT_DEFINED","n_queries":0}
                continue
            success=np.array([sum(rank==1 for rank in v) for v in values])
            rng=np.random.default_rng(SEED)
            dist=[]
            for i in range(0,B,64):
                ix=rng.integers(0,len(rs),size=(min(64,B-i),len(rs)))
                denominator=counts[ix].sum(1)
                ok=denominator>0
                dist.extend((success[ix].sum(1)[ok]/denominator[ok]).tolist())
            ranks=[rank for v in values for rank in v]
            result[name]={"n_queries":total,"n_questions":int((counts>0).sum()),"R@1":float(success.sum()/total),
                          "R@1_clustered_ci95":[float(v) for v in np.percentile(dist,[2.5,97.5])],
                          "R@3":float(np.mean(np.array(ranks)<=3)),"R@5":float(np.mean(np.array(ranks)<=5)),"MRR":float(np.mean(1/np.array(ranks)))}
        output[group]=result
    return output
