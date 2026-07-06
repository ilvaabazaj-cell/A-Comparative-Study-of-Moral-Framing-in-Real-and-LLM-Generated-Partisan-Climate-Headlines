#!/usr/bin/env python3
# =============================================================================
#  Moral Framing of Partisan Climate Discourse:
#  Real News Titles versus a Matched LLM-Generated Corpus
#
#  Reproduces every number reported in the paper (Tables 1-2, effect sizes,
#  confidence intervals, the LibertyMFD robustness check, and the log-odds frame).
#
#  Inputs (in ./data/):
#     earth_day_real_articles-2.json   real titles (Media Cloud, quintiles Q1-Q5)
#     dem_titles_final.json            synthetic Democrat titles (Claude Sonnet 4.6)
#     rep_titles_final.json            synthetic Republican titles (Claude Sonnet 4.6)
#  Downloaded automatically if absent:
#     emfd.csv                         eMFD (Hopp et al. 2021), MIT-licensed
#     libertymfd_v1/2/3                LibertyMFD (Araque et al. 2022), LGPL-3.0
#
#  Run:  python analysis.py
# =============================================================================
import json, re, math, collections, os, urllib.request
import numpy as np, pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as sw

SEED = 42
np.random.seed(SEED)
DATA, OUT = "data", "outputs"
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

for pkg in ["wordnet", "omw-1.4", "averaged_perceptron_tagger_eng", "stopwords"]:
    try: nltk.download(pkg, quiet=True)
    except Exception: pass

# --- download third-party dictionaries if absent -----------------------------
EMFD_URL = "https://raw.githubusercontent.com/medianeuroscience/emfdscore/master/emfdscore/dictionaries/emfd_scoring.csv"
LIB_URLS = {
    "v1": "https://raw.githubusercontent.com/oaraque/moral-foundations/master/liberty/1st_version/lexicon_filtered.tsv",
    "v2": "https://raw.githubusercontent.com/oaraque/moral-foundations/master/liberty/2nd_version/we_lexicon_final.csv",
    "v3": "https://raw.githubusercontent.com/oaraque/moral-foundations/master/liberty/3rd_version/liberty_moral_lexicon.tsv",
}
def fetch(url, path):
    if not os.path.exists(path):
        print(f"downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
fetch(EMFD_URL, f"{DATA}/emfd.csv")
for k, u in LIB_URLS.items():
    fetch(u, f"{DATA}/libertymfd_{k}." + ("tsv" if u.endswith("tsv") else "csv"))

# --- preprocessing -----------------------------------------------------------
STOP = set(sw.words("english")) | {"say","new","could","would","one","us","day","via","first","show"}
KEEP = {"NN","NNS","NNP","NNPS","JJ","JJR","JJS","VB","VBD","VBG","VBN","VBP","VBZ","RB"}
LEM = WordNetLemmatizer()
_wp = lambda t: {"N":"n","V":"v","J":"a","R":"r"}.get(t[0], "n")
def tok(title):
    t = title.lower().replace("\u2019","'").replace("\u2014"," ").replace("\u2013"," ")
    t = re.sub(r"'s\b", "", t)
    out = []
    for w, tag in nltk.pos_tag(re.findall(r"[a-z]+", t)):
        if tag in KEEP:
            l = LEM.lemmatize(w, _wp(tag))
            if l not in STOP and len(l) >= 3:
                out.append(l)
    return out

def load_real(path):
    raw = json.load(open(path))
    def grp(quints, year):
        seen, docs = set(), []
        for q in quints:
            for a in raw.get(f"{q}_22_April_{year}", []):
                if a.get("language", "en") != "en": continue
                t = a["title"].strip()
                if t and t.lower() not in seen:
                    seen.add(t.lower()); docs.append(tok(t))
        return docs
    dem = [d for y in (2016, 2017) for d in grp(["Q1", "Q2"], y)]
    rep = [d for y in (2016, 2017) for d in grp(["Q4", "Q5"], y)]
    return dem, rep
def load_syn(path):
    d = json.load(open(path))
    return [tok(a["title"]) for y in ["2016-04-22", "2017-04-22"] for a in d["articles"][y]]

REAL_DEM, REAL_REP = load_real(f"{DATA}/earth_day_real_articles-2.json")
SYN_DEM = load_syn(f"{DATA}/dem_titles_final.json")
SYN_REP = load_syn(f"{DATA}/rep_titles_final.json")
GROUPS = {"realDem": REAL_DEM, "realRep": REAL_REP, "synDem": SYN_DEM, "synRep": SYN_REP}

# --- statistics --------------------------------------------------------------
def permutation_p(a, b, n=20000):
    obs = a.mean() - b.mean(); pool = np.concatenate([a, b]); na = len(a); c = 0
    for _ in range(n):
        np.random.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= abs(obs): c += 1
    return obs, (c + 1) / (n + 1)
def cohen_d(a, b):
    na, nb = len(a), len(b)
    s = math.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return (a.mean()-b.mean())/s if s > 0 else 0.0
def boot_ci_d(a, b, n=3000):
    ds = [cohen_d(a[np.random.randint(0,len(a),len(a))], b[np.random.randint(0,len(b),len(b))]) for _ in range(n)]
    return np.percentile(ds, [2.5, 97.5])

LOG = []
def out(m=""): print(m); LOG.append(str(m))

# --- eMFD --------------------------------------------------------------------
EMFD = pd.read_csv(f"{DATA}/emfd.csv").set_index("word")
FOUND = ["care", "fairness", "loyalty", "authority", "sanctity"]
PROB = {w: EMFD.loc[w, [f+"_p" for f in FOUND]].values.astype(float) for w in EMFD.index}
def emfd_matrix(docs):
    return np.array([np.mean([PROB[w] for w in d if w in PROB], 0)
                     for d in docs if any(w in PROB for w in d)])
M = {k: emfd_matrix(v) for k, v in GROUPS.items()}

out("CORPUS SIZES and eMFD coverage (matched moral words / title)")
for k, v in GROUPS.items():
    out(f"  {k:<8} N={len(v):<4} coverage={np.mean([sum(w in PROB for w in d) for d in v if d]):.1f}")

out("\nTABLE 1  mean eMFD foundation probability + individualizing/binding")
rows = []
for i, f in enumerate(FOUND):
    r = {"foundation": f, **{k: round(M[k][:, i].mean(), 3) for k in GROUPS}}
    rows.append(r)
    out(f"  {f:<10}" + "".join(f"{r[k]:>9.3f}" for k in ['realDem','realRep','synDem','synRep']))
for name, idx in [("individualizing",[0,1]), ("binding",[2,3,4])]:
    rows.append({"foundation": name, **{k: round(M[k][:, idx].sum(1).mean(), 3) for k in GROUPS}})
rows.append({"foundation":"IB_ratio", **{k: round(M[k][:,[0,1]].sum(1).mean()/M[k][:,[2,3,4]].sum(1).mean(),3) for k in GROUPS}})
pd.DataFrame(rows).to_csv(f"{OUT}/table1_moral_means.csv", index=False)

out("\nREAL partisan contrast (Dem vs Rep): Cohen d with 95% bootstrap CI")
es = []
for i, f in enumerate(FOUND):
    d = cohen_d(M["realDem"][:, i], M["realRep"][:, i])
    lo, hi = boot_ci_d(M["realDem"][:, i], M["realRep"][:, i])
    _, p = permutation_p(M["realDem"][:, i], M["realRep"][:, i])
    es.append((f, round(d,2), round(lo,2), round(hi,2), round(p,3)))
    out(f"  {f:<10} d={d:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  p={p:.3f}")
pd.DataFrame(es, columns=["foundation","d","ci_lo","ci_hi","p"]).to_csv(f"{OUT}/real_partisan_effectsizes.csv", index=False)

out("\nTABLE 2  fidelity: partisan-gap p (real/synth) + inflation d (real vs synth)")
fid = []
for i, f in enumerate(FOUND):
    _, pr = permutation_p(M["realDem"][:, i], M["realRep"][:, i], 10000)
    _, ps = permutation_p(M["synDem"][:, i], M["synRep"][:, i], 10000)
    dd = cohen_d(M["realDem"][:, i], M["synDem"][:, i])
    dr = cohen_d(M["realRep"][:, i], M["synRep"][:, i])
    fid.append((f, round(pr,3), round(ps,3), round(dd,2), round(dr,2)))
    out(f"  {f:<10} gap_p real={pr:.3f} synth={ps:.3f} | inflation_d Dem={dd:+.2f} Rep={dr:+.2f}")
pd.DataFrame(fid, columns=["foundation","gap_p_real","gap_p_synth","inflation_d_dem","inflation_d_rep"]).to_csv(f"{OUT}/table2_fidelity.csv", index=False)

# --- LibertyMFD (sixth foundation) robustness check --------------------------
def load_lib(v):
    if v == "v1":
        d = pd.read_csv(f"{DATA}/libertymfd_v1.tsv", sep="\t"); return dict(zip(d["word"], d["valence"]))
    if v == "v2":
        d = pd.read_csv(f"{DATA}/libertymfd_v2.csv"); return dict(zip(d["word"], d["score"]))
    d = pd.read_csv(f"{DATA}/libertymfd_v3.tsv", sep="\t", header=0, names=["word","score"]); return dict(zip(d["word"], d["score"]))
def lib_score(docs, LIB):
    return np.array([np.mean([LIB[w] for w in d if w in LIB]) for d in docs if any(w in LIB for w in d)])

out("\nLIBERTY foundation (LibertyMFD, Araque et al. 2022) - robustness across 3 released versions")
lib_rows = []
for v in ["v1", "v2", "v3"]:
    LIB = load_lib(v)
    rd, rr = lib_score(REAL_DEM, LIB), lib_score(REAL_REP, LIB)
    sd, sr = lib_score(SYN_DEM, LIB), lib_score(SYN_REP, LIB)
    _, pr = permutation_p(rr, rd); _, ps = permutation_p(sr, sd)
    lib_rows.append((v, len(LIB), round(cohen_d(rr, rd),2), round(pr,3), round(cohen_d(sr, sd),2), round(ps,3)))
    out(f"  {v} (size {len(LIB)}): REAL d={cohen_d(rr,rd):+.2f} p={pr:.3f} | SYNTH d={cohen_d(sr,sd):+.2f} p={ps:.3f}")
pd.DataFrame(lib_rows, columns=["version","size","real_d","real_p","synth_d","synth_p"]).to_csv(f"{OUT}/liberty_robustness.csv", index=False)

# --- weighted log-odds (Monroe et al. 2008): real Dem vs Rep -----------------
def counts(docs):
    c = collections.Counter()
    for d in docs: c.update(d)
    return c
def log_odds(cA, cB, a0=500.0):
    vocab = set(cA) | set(cB); bg = collections.Counter(); bg.update(cA); bg.update(cB)
    nbg = sum(bg.values()); al = {w: a0*bg[w]/nbg for w in vocab}
    nA, nB, aT = sum(cA.values()), sum(cB.values()), sum(al.values()); r = []
    for w in vocab:
        lA = math.log((cA[w]+al[w])/(nA+aT-cA[w]-al[w])); lB = math.log((cB[w]+al[w])/(nB+aT-cB[w]-al[w]))
        r.append((w, cA[w], cB[w], (lA-lB)/math.sqrt(1/(cA[w]+al[w])+1/(cB[w]+al[w]))))
    return pd.DataFrame(r, columns=["word","n_dem","n_rep","z"]).sort_values("z")
lo_real = log_odds(counts(REAL_DEM), counts(REAL_REP))
lo_syn = log_odds(counts(SYN_DEM), counts(SYN_REP))
lo_real.to_csv(f"{OUT}/logodds_real.csv", index=False)
def side(df, n, pos): return set(df.tail(n)["word"]) if pos else set(df.head(n)["word"])
out("\nFRAME OVERLAP real vs synthetic (top-25 Jaccard)")
for lab, pos in [("Democrat", True), ("Republican", False)]:
    a, b = side(lo_real, 25, pos), side(lo_syn, 25, pos)
    out(f"  {lab:<11} Jaccard = {len(a & b)/len(a | b):.2f}")
out("Real Republican-audience frame: " + ", ".join(lo_real.head(12)["word"]))
out("Real Democrat-audience frame:   " + ", ".join(lo_real.tail(12).iloc[::-1]["word"]))

open(f"{OUT}/summary.txt", "w").write("\n".join(LOG))
out("\n[done] all tables written to outputs/")
