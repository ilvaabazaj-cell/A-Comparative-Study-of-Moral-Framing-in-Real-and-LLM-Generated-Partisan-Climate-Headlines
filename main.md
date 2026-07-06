# Introduction

Climate change is one of the most polarised issues in US politics, and a
growing literature uses text analysis to describe how partisan media
frame it. Within this project, an emotion-based analysis with EmoAtlas
\[2\] reports that language models overstate the partisan signal found
in real coverage. Emotion is only one layer of framing, however. In
Entman's terms \[1\], framing is about which concepts and values an
issue is tied to, and the standard cognitive account of the value layer
is Moral Foundations Theory (MFT). MFT distinguishes the individualising
foundations (care, fairness) from the binding ones (loyalty, authority,
sanctity), and holds that the political left draws more on the former
and the right more on the latter \[3\]. A sixth foundation, liberty, was
proposed later \[4\]. This left--right contrast is the assumption behind
moral-reframing appeals in climate communication \[5, 6\], yet it is
rarely tested on the partisan discourse itself.

Two primary questions are addressed. First, how morally polarised is
real partisan climate discourse, and does the MFT contrast actually
appear in it? Second, does a size-matched, LLM-generated corpus
reproduce whatever moral structure the real data has? The second
question speaks to the use of language models as "silicon samples" of
human groups, whose usefulness depends on their algorithmic fidelity
\[10\]. The present study differs from the emotion analysis in what it
measures (moral foundations rather than affect), in method
(between-group tests with effect sizes), and in taking the real corpus,
not the model, as its primary object.

# Data and Methods

## Corpora

Real titles were collected through the Media Cloud API by querying
climate keywords (climate change/crisis/action/emergency, global
warming) on 22 April 2016 and 2017 across Media Cloud's
partisan-audience quintile collections, where an outlet's quintile
reflects the partisanship of the audience that shares it. Following the
project protocol, quintiles Q1--Q2 are treated as the Democrat-leaning
group and Q4--Q5 as the Republican-leaning group, and remove duplicate
titles. This gives Democrat $N=362$ (178 in 2016, 184 in 2017) and
Republican $N=113$ (53, 60).

## Corpus generation

Synthetic titles are a size-matched corpus (363/113) written by Claude
Sonnet 4.6 (Anthropic) from a single fixed prompt per party. The prompt
gave the date, the required count, the topical keywords above, the
partisan alignment (what Democrat- or Republican-leaning voters would
have shared), and an instruction to invent titles rather than retrieve
real ones. The full prompt and data are available in the project
repository [@github_repo].

## Measures

Titles were lowercased, POS-tagged and lemmatised (NLTK), keeping
content words. Moral content is scored with the extended Moral
Foundations Dictionary (eMFD) \[7\], which gives each word continuous
probabilities of expressing the five foundations; a title's score is the
mean over its matched words. Titles are short (mean 3.4 matched words
for real, 4.6--5.1 for synthetic), hence, a single measure is not solely
relied upon. The sixth foundation is added using the validated
LibertyMFD lexicon \[8\], scoring each title as the mean liberty valence
over its matched words, and all three released versions of that lexicon
are reported as a robustness check. Separately, weighted log-odds with
an informative Dirichlet prior \[9\] recover each group's characteristic
vocabulary, and the Jaccard overlap of the top-25 partisan terms
measures how far the real and synthetic frames coincide. Group contrasts
use two-sided permutation tests ($2 \times 10^4$ resamples) with Cohen's
$d$, 95% bootstrap confidence intervals, and Bonferroni correction
across the five eMFD foundations ($\alpha=.01$).

<figure id="fig:moral_foundations" data-latex-placement="t">
<img src="./figure1_real_vs_synth.png" style="width:68.0%" />
<figcaption>Mean eMFD foundation probabilities by party, with bootstrap
95% confidence intervals. In the real corpus (left) the Democrat and
Republican profiles almost coincide; in the synthetic corpus (right) the
partisan gap widens (fairness and loyalty reach significance, "**") and
care, authority and sanctity sit well above their real
levels.</figcaption>
</figure>

# Results

## Real partisan discourse is barely moralised at the foundation level

Table 1 describes the real corpus. No eMFD foundation survives
Bonferroni correction. More than that, the effects are bounded small:
every 95% bootstrap confidence interval falls within $|d| < 0.5$, the
largest being fairness ($d=0.23$, CI $[-0.45, -0.02]$). The MFT contrast
does not appear. Democrats are not more individualising---if anything
the Republican-leaning corpus is marginally more so ($p=.051$)---the
binding foundations do not differ ($p=.28$), and the
individualising/binding ratio is nearly identical across parties (0.70
vs 0.73).

+-----------------+---------------------------+-------------------------+
|                 | **Real (Media Cloud)**    | **Synthetic (LLM)**     |
+:================+:===========:+:===========:+:==========:+:==========:+
| 2-3 (lr)4-5     | Dem         | Rep         | Dem        | Rep        |
| **Foundation**  |             |             |            |            |
+-----------------+-------------+-------------+------------+------------+
| Care            | .101        | .107        | .125       | .123       |
+-----------------+-------------+-------------+------------+------------+
| Fairness        | .088        | .096        | .091       | .098       |
+-----------------+-------------+-------------+------------+------------+
| Loyalty         | .092        | .096        | .090       | .097       |
+-----------------+-------------+-------------+------------+------------+
| Authority       | .091        | .094        | .102       | .102       |
+-----------------+-------------+-------------+------------+------------+
| Sanctity        | .088        | .089        | .093       | .094       |
+-----------------+-------------+-------------+------------+------------+
| Individualising | .189        | .203        | .215       | .222       |
+-----------------+-------------+-------------+------------+------------+
| Binding         | .270        | .279        | .285       | .293       |
+-----------------+-------------+-------------+------------+------------+
| I/B ratio       | .700        | .730        | .755       | .758       |
+-----------------+-------------+-------------+------------+------------+

: Mean eMFD foundation probability by corpus, with the individualising
(care+fairness) and binding (loyalty+authority+sanctity) composites.

The sixth foundation behaves the same way: across the three LibertyMFD
versions the real partisan difference ranges from $d=0.10$ to $d=0.34$
and reaches significance only in the smallest lexicon, hence it is not
interpreted as a reliable effect. On every validated moral instrument
applied, the two audiences look alike.

<figure id="fig:log_odds" data-latex-placement="t">
<img src="./figure2_real_frame.png" style="width:72.0%" />
<figcaption>Real partisan vocabulary (weighted log-odds). This wording
is almost disjoint from the synthetic corpus’s (top-25 Jaccard <span
class="math inline"> ≤ .09</span>).</figcaption>
</figure>

## The partisan difference is topical, not moral

Where the foundations are quiet, vocabulary is not. The
Democrat-audience frame is built around the treaty and its
politics---*Paris, accord, policy, deal, Washington*---with some noise
from the day's other news (*prince*, the musician who had just died).
The Republican-audience frame turns on scepticism and
opposition---*skeptic, wrong, liberal, protester*---with an economic
thread (*cost*). This is a difference in what the two sides talk about,
not in the moral register they use to talk about it, and it is exactly
what the foundation scores miss.

+--------------------+-------------------------+-----------------------+
|                    | **partisan gap $p$**    | **inflation $d$**     |
+:===================+:==========:+:==========:+:=========:+:=========:+
| 2-3 (lr)4-5        | real       | synth      | Dem       | Rep       |
| **Foundation**     |            |            |           |           |
+--------------------+------------+------------+-----------+-----------+
| Care               | .18        | .69        | $-0.68$   | $-0.45$   |
+--------------------+------------+------------+-----------+-----------+
| Fairness           | .03        | **.002**   | $-0.10$   | $-0.08$   |
+--------------------+------------+------------+-----------+-----------+
| Loyalty            | .24        | **.002**   | $+0.09$   | $-0.06$   |
+--------------------+------------+------------+-----------+-----------+
| Authority          | .32        | .92        | $-0.40$   | $-0.26$   |
+--------------------+------------+------------+-----------+-----------+
| Sanctity           | .68        | .82        | $-0.26$   | $-0.20$   |
+--------------------+------------+------------+-----------+-----------+

: Fidelity diagnostics. Partisan gap p: is the Dem--Rep difference
significant within each corpus? Inflation d: real-vs-synthetic effect
size within party (negative = synthetic higher).

## The LLM over-moralises and over-polarises

The synthetic corpus departs from the real one in two consistent ways
(Table 2). First, it raises the overall moral load: care, authority and
sanctity are all higher in synthetic than in real titles, for both
parties (care $d=0.68$ Dem / $0.45$ Rep; authority $0.40/0.26$; sanctity
$0.26/0.20$; all $p < .001$ except Republican authority, $p=.056$).
Second, it sharpens the partisan contrast: the weak real fairness gap
and the non-significant real loyalty gap both become significant in the
synthetic data ($p=.002$), so the model produces a moral divide that the
real corpus does not support. The two frames also share little
vocabulary---the top-25 partisan terms overlap at Jaccard .09 (Democrat)
and .02 (Republican). The model captures the topic of partisan climate
coverage but neither its muted moral profile nor its real wording.

# Discussion

The moral distance between Democrat- and Republican-audience climate
titles is small. Neither the five eMFD foundations nor the liberty
foundation reliably tells the two groups apart, so the premise behind
moral-reframing communication---that the sides speak in different
foundational registers \[5, 6\]---is not visible in the headlines they
circulate. The partisan signal that does exist is topical: a
treaty-and-policy frame on one side, a scepticism-and-cost frame on the
other.

The synthetic corpus tells a different and, for present purposes, a more
informative story. Benchmarked against real data, the model is reliably
more moralised and more polarised than the discourse it imitates. It
inflates three of the five foundations and manufactures partisan gaps in
the other two. A plausible reading is that a model asked to write "what
a partisan would share" produces the prototype of each side rather than
its ordinary output, and prototypes are cleaner and more moralised than
everyday headlines. Either way, the practical lesson is concrete: using
generated text as a proxy for real partisan discourse \[10\] will
overstate both how moral and how divided that discourse is, and
synthetic corpora should be checked against real data before they are
trusted.

**Limitations.** The real corpus covers two single days and carries
their topical noise (a celebrity death, the 2016 campaign), and the
Republican sample is small ($N=113$), so the null partisan result is
reported as bounded, not proven; the inferential weight rests on the
well-powered real-versus-synthetic comparisons. Titles are short, so
eMFD leans on few words, and LibertyMFD's coverage is lower still (2--6
words per title), which is one reason its estimate is unstable across
versions. The synthetic corpus is a single generation from one model and
prompt, so the magnitudes reported here are specific to it.

# Conclusion

This study demonstrates that real partisan climate discourse in media
headlines is primarily distinguished by topical framing rather than
fundamental moral divergence. Contrary to the assumptions of
moral-reframing theories, Democrat- and Republican-leaning audiences
circulate coverage with remarkably similar moral profiles. However,
synthetic corpora generated by Large Language Models fail to capture
this empirical reality, instead producing over-moralised and
artificially polarised text. These findings issue a clear methodological
warning for computational social science: while LLMs are powerful tools
for text generation, treating them as unverified proxies for human
groups risks amplifying polarisation effects that do not exist in the
field. Future work should continue to establish rigorous empirical
benchmarks before relying on synthetic data for political and social
analysis.

::: thebibliography
11

Entman, R.M. (1993). Framing: Toward clarification of a fractured
paradigm. *Journal of Communication*, 43(4), 51-58. Ruffo, G., & Stella,
M. (2025). EmoAtlas: An emotional network analyzer of texts. *Behavior
Research Methods*, 57, 77. Graham, J., Haidt, J., & Nosek, B.A. (2009).
Liberals and conservatives rely on different sets of moral foundations.
*Journal of Personality and Social Psychology*, 96(5), 1029-1046. Iyer,
R., Koleva, S., Graham, J., Ditto, P., & Haidt, J. (2012). Understanding
libertarian morality. *PLoS ONE*, 7(8), e42366. Feinberg, M., & Willer,
R. (2013). The moral roots of environmental attitudes. *Psychological
Science*, 24(1), 56-62. Wolsko, C., Ariceaga, H., & Seiden, J. (2016).
Red, white, and blue enough to be green: moral framing effects on
climate attitudes. *Journal of Experimental Social Psychology*, 65,
7-19. Hopp, F.R., Fisher, J.T., Cornell, D., Huskey, R., & Weber, R.
(2021). The extended Moral Foundations Dictionary (eMFD). *Behavior
Research Methods*, 53(1), 232-246. Araque, O., Gatti, L., & Kalimeri, K.
(2022). LibertyMFD: a lexicon to assess the moral foundation of liberty.
*Behavior Research Methods*, 54(5), 2307-2320. Monroe, B.L., Colaresi,
M.P., & Quinn, K.M. (2008). Fightin' words: Lexical feature selection
and evaluation for identifying the content of political conflict.
*Political Analysis*, 16(4), 372-403. Argyle, L.P., Busby, E.C., Fulda,
N., Gubler, J.R., Rytting, C., & Wingate, D. (2023). Out of one, many:
Using language models to simulate human samples. *Political Analysis*,
31(3), 337-351. Abazaj, I. (2026). A Comparative Study of Moral Framing
in Real and LLM-Generated Partisan Climate Headlines. GitHub Repository,
<https://github.com/ilvaabazaj-cell/A-Comparative-Study-of-Moral-Framing-in-Real-and-LLM-Generated-Partisan-Climate-Headlines>.
:::
