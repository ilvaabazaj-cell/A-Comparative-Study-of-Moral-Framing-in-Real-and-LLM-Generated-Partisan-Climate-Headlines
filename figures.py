#!/usr/bin/env python3
# =============================================================================
#  Figures for "Moral and Economic Framing of Partisan Climate Discourse".
#  Produces:  outputs/figure1_real_vs_synth.png   (two-panel moral profiles)
#             outputs/figure2_real_frame.png       (real log-odds partisan frame)
#  Run after / alongside analysis.py:  python figures.py
# =============================================================================
import json, re, math, collections, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as sw

np.random.seed(7)
DATA, OUT = "data", "outputs"
os.makedirs(OUT, exist_ok=True)
for pkg in ["wordnet", "omw-1.4", "averaged_perceptron_tagger_eng", "stopwords"]:
    try: nltk.download(pkg, quiet=True)
    except Exception: pass

# --- preprocessing (identical to analysis.py) ------------------------------
STOP = set(sw.words("english")) | {"say","new","could","would","one","us","day","via","first","show"}
KEEP = {"NN","NNS","NNP","NNPS","JJ","JJR","JJS","VB","VBD","VBG","VBN","VBP","VBZ","RB"}
LEM = WordNetLemmatizer(); _wp = lambda t: {"N":"n","V":"v","J":"a","R":"r"}.get(t[0], "n")
def tok(title):
    t = title.lower().replace("\u2019","'").replace("\u2014"," ").replace("\u2013"," ")
    t = re.sub(r"'s\b","",t); out=[]
    for w,tag in nltk.pos_tag(re.findall(r"[a-z]+",t)):
        if tag in KEEP:
            l=LEM.lemmatize(w,_wp(tag))
            if l not in STOP and len(l)>=3: out.append(l)
    return out
def load_real(p):
    raw=json.load(open(p))
    def g(qs,y):
        seen,d=set(),[]
        for q in qs:
            for a in raw.get(f"{q}_22_April_{y}",[]):
                if a.get("language","en")!="en": continue
                t=a["title"].strip()
                if t and t.lower() not in seen: seen.add(t.lower()); d.append(tok(t))
        return d
    return ([x for y in(2016,2017) for x in g(["Q1","Q2"],y)],
            [x for y in(2016,2017) for x in g(["Q4","Q5"],y)])
def load_syn(p):
    d=json.load(open(p)); return [tok(a["title"]) for y in ["2016-04-22","2017-04-22"] for a in d["articles"][y]]

REAL_DEM,REAL_REP=load_real(f"{DATA}/earth_day_real_articles-2.json")
SYN_DEM=load_syn(f"{DATA}/dem_titles_final.json"); SYN_REP=load_syn(f"{DATA}/rep_titles_final.json")
EMFD=pd.read_csv(f"{DATA}/emfd.csv").set_index("word")
FOUND=["care","fairness","loyalty","authority","sanctity"]
PROB={w:EMFD.loc[w,[f+"_p" for f in FOUND]].values.astype(float) for w in EMFD.index}
def mat(docs):
    r=[np.mean([PROB[w] for w in d if w in PROB],0) for d in docs if any(w in PROB for w in d)]
    return np.array(r)
MRD,MRR,MSD,MSR=mat(REAL_DEM),mat(REAL_REP),mat(SYN_DEM),mat(SYN_REP)

DEM,REP="#2c6fbb","#c1332e"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":160})
def boot(X,n=4000):
    idx=np.arange(len(X)); m=np.array([X[np.random.choice(idx,len(idx),True)].mean(0) for _ in range(n)])
    return X.mean(0),np.percentile(m,2.5,0),np.percentile(m,97.5,0)

# --- Figure 1: two panels real | synthetic ---------------------------------
rd,ld,hd=boot(MRD); rr,lr,hr=boot(MRR); sd,lsd,hsd=boot(MSD); sr,lsr,hsr=boot(MSR)
fig,ax=plt.subplots(1,2,figsize=(9.2,3.9),sharey=True); x=np.arange(len(FOUND)); w=0.38
for a,(md,lo_d,hi_d,mr,lo_r,hi_r,ttl) in zip(ax,[
        (rd,ld,hd,rr,lr,hr,"REAL titles (Media Cloud)"),
        (sd,lsd,hsd,sr,lsr,hsr,"SYNTHETIC titles (LLM)")]):
    a.bar(x-w/2,md,w,yerr=[md-lo_d,hi_d-md],color=DEM,alpha=.85,capsize=2,label="Democrat")
    a.bar(x+w/2,mr,w,yerr=[mr-lo_r,hi_r-mr],color=REP,alpha=.85,capsize=2,label="Republican")
    a.set_xticks(x); a.set_xticklabels([f.capitalize() for f in FOUND],rotation=25,ha="right",fontsize=8.5)
    a.set_title(ttl,fontsize=10,fontweight="bold"); a.set_ylim(0,0.16)
ax[0].set_ylabel("mean eMFD foundation probability"); ax[0].legend(frameon=False,fontsize=9)
for f in ["fairness","loyalty"]:
    ax[1].text(FOUND.index(f),0.108,"**",ha="center",fontsize=11,fontweight="bold")
fig.suptitle("Moral-foundations profiles by party: real (Media Cloud) vs. synthetic (LLM) titles",
             fontsize=10.5,fontweight="bold",y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/figure1_real_vs_synth.png",bbox_inches="tight")
print("figure1 written")

# --- Figure 2: real partisan lexical frame (log-odds) ----------------------
def counts(docs):
    c=collections.Counter()
    for d in docs: c.update(d)
    return c
def log_odds(cA,cB,a0=500.0):
    vocab=set(cA)|set(cB); bg=collections.Counter(); bg.update(cA); bg.update(cB)
    nbg=sum(bg.values()); al={w:a0*bg[w]/nbg for w in vocab}
    nA,nB,aT=sum(cA.values()),sum(cB.values()),sum(al.values()); rows=[]
    for w in vocab:
        lA=math.log((cA[w]+al[w])/(nA+aT-cA[w]-al[w])); lB=math.log((cB[w]+al[w])/(nB+aT-cB[w]-al[w]))
        rows.append((w,(lA-lB)/math.sqrt(1/(cA[w]+al[w])+1/(cB[w]+al[w]))))
    return pd.DataFrame(rows,columns=["word","z"]).sort_values("z")
lo=log_odds(counts(REAL_DEM),counts(REAL_REP))
topD=lo.tail(11); topR=lo.head(11).iloc[::-1]
fig,ax=plt.subplots(figsize=(8.4,4.2))
yD=np.arange(len(topD))+len(topR)+1; yR=np.arange(len(topR))
ax.barh(yR,topR["z"],color=REP,alpha=.85); ax.barh(yD,topD["z"],color=DEM,alpha=.85)
for yi,(_,r) in zip(yR,topR.iterrows()): ax.text(-0.03,yi,r["word"],ha="right",va="center",fontsize=9)
for yi,(_,r) in zip(yD,topD.iterrows()): ax.text(0.03,yi,r["word"],ha="left",va="center",fontsize=9)
ax.axvline(0,color="k",lw=.8); ax.set_yticks([]); ax.set_xlabel("weighted log-odds z-score (Monroe et al., 2008)")
ax.text(-2.2,yR.mean(),"Republican-audience\nframe (real)",ha="center",va="center",color=REP,fontweight="bold",fontsize=9.5)
ax.text(1.9,yD.mean(),"Democrat-audience\nframe (real)",ha="center",va="center",color=DEM,fontweight="bold",fontsize=9.5)
ax.set_xlim(-3.2,3.0)
ax.set_title("Real partisan lexical frame (Media Cloud titles, pooled 2016\u20132017)",fontsize=10.5,fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/figure2_real_frame.png",bbox_inches="tight")
print("figure2 written")
