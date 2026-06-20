# Mathematical Reference (v2 summary)

> This root file has been reconciled to the implemented build. The full,
> authoritative reference is [`docs/mathematics.md`](docs/mathematics.md); this
> page is a short summary of the equations as they exist in `src/`.

All distance-like matrices stay symmetric, zero-diagonal, non-negative
($M=M^\top$, $M_{ii}=0$, $M_{ij}\ge0$).

## Distance

$$
H=\frac{1}{L_{ij}}\sum_{k}\mathbf{1}[s_i[k]\ne s_j[k]],\quad
P=\frac{|L_i-L_j|}{\max(L_i,L_j)},\quad
d=\alpha H+(1-\alpha)P\ \ (\alpha=0.9,\ d\in[0,1]).
$$

Both empty $\Rightarrow0$; exactly one empty $\Rightarrow1$.

## Four-point violation

$$
S_1=X_{ij}+X_{kl},\ S_2=X_{ik}+X_{jl},\ S_3=X_{il}+X_{jk},\quad
\delta=\max(S)-\mathrm{secondmax}(S)\ge0.
$$

`tropical_score` reports mean/median/max violation, `l2_loss`$=\sum\delta^2$, and
`percent_exact`$=100\cdot\#\{\delta<\varepsilon\}/n_{\text{sampled}}$ (a percentage).

## Correction and optimizer

$X=D+\omega$; objective $\mathcal{L}(\omega)=\sum_q\delta_q^2+\lambda\| w\|_2^2$
(sum form; $w$ is the upper-triangular vector of $\omega$). Practical subgradient:
$+2\delta$ on the max group's two pair entries, $-2\delta$ on the second group's,
plus $2\lambda w$.

**Hybrid optimizer** — normalized tropical-norm step (never Euclidean):

$$
w\leftarrow w-\frac{\gamma}{\| g\|_{\mathrm{tr}}+\varepsilon}\,g,\qquad
\| g\|_{\mathrm{tr}}=\max(g)-\min(g),
$$

then project $X=\Pi_{\mathcal{D}_n}(D+\omega)$ (symmetrize, clip $\ge0$, zero
diagonal). Stop when $\| g\|_{\mathrm{tr}}\le\varepsilon$; return the best
iterate (so $X$ is never worse than $D$). The literal
$\eta=\gamma\| g\|_{\mathrm{tr}}/(\sqrt{p}+\varepsilon)$ diverges and is
**not** used (see [`docs/mathematics.md`](docs/mathematics.md)).

## Relative improvement and Neighbor-Joining

$\text{RI}=(L_2(D)-L_2(X))/(L_2(D)+\varepsilon)$ on a shared quadruplet sample.
Neighbor-Joining uses $Q_{ij}=(m-2)M_{ij}-r_i-r_j$ with the standard branch-length
and reduction formulas; the tree is exported as Newick, edge list, and DOT.

Reducing four-point violations improves compatibility with an additive tree
metric; it does not establish biological correctness.
