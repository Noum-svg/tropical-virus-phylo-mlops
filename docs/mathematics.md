# Mathematics (as implemented)

This document describes the math exactly as implemented in `src/`. Where the
implementation deliberately departs from an earlier draft spec, the reason is
stated in an **Implementation note**.

All matrices are kept symmetric, zero-diagonal, and (for distance-like matrices)
non-negative:

$$
\mathcal{D}_n = \{ M \in \mathbb{R}^{n\times n} \mid M = M^\top,\; M_{ii}=0,\; M_{ij}\ge 0 \}.
$$

## 1. Sequence cleaning

For each raw sequence: uppercase → remove whitespace → replace `U` with `T` →
keep only `A,C,G,T`. Rows with an empty name, a cleaned length below
`min_seq_length`, or a duplicate `(virus_name, clean_sequence)` are dropped.
(`src/data_loader.py`)

## 2. Pairwise distance and matrix $D$

For cleaned sequences $s_i, s_j$ with lengths $L_i, L_j$ and
$L_{ij}=\min(L_i,L_j)$:

$$
H = \frac{1}{L_{ij}}\sum_{k=0}^{L_{ij}-1}\mathbf{1}[s_i[k]\ne s_j[k]],
\qquad
P = \frac{|L_i-L_j|}{\max(L_i,L_j)},
\qquad
d = \alpha H + (1-\alpha) P .
$$

Edge cases: both empty $\Rightarrow 0$; exactly one empty $\Rightarrow 1$.
Default $\alpha=0.9$, which keeps $d\in[0,1]$. (`src/distances.py`)

> **Implementation note.** The convex combination $\alpha H+(1-\alpha)P$ is used
> (not the unweighted $H+P$) so that $D\in[0,1]$ as required by the build spec
> and its tests. $\alpha$ is configurable in `params.yaml`.

## 3. Tropical four-point violation

For a quadruplet $i<j<k<l$ of a symmetric matrix $X$:

$$
S_1 = X_{ij}+X_{kl},\quad S_2 = X_{ik}+X_{jl},\quad S_3 = X_{il}+X_{jk},
$$

$$
\delta_{ijkl}(X) = \max(S_1,S_2,S_3) - \mathrm{secondmax}(S_1,S_2,S_3) \ge 0 .
$$

$\delta=0$ (within $\varepsilon$) iff the quadruplet is tree-compatible (the max
is attained at least twice). `tropical_score` aggregates `mean/median/max`
violation, `l2_loss` $=\sum\delta^2$, and `percent_exact` $=100\cdot\#\{\delta<\varepsilon\}/n_{\text{sampled}}$.
For large $n$, quadruplets are sampled without replacement with a fixed seed.
(`src/tropical_grassmannian.py`)

## 4. Correction, loss, and subgradient

Model $X = D + \omega$ with $\omega$ symmetric and zero-diagonal, stored as the
upper-triangular vector $w$ of length $p=n(n-1)/2$. Let $\mathcal{Q}$ denote
the evaluated quadruplets. The objective is:

$$
L(w) = \sum_{q\in\mathcal{Q}} \delta_q(D+\omega)^2
       + \lambda_{\text{reg}}\sum_r w_r^2 .
$$

Subgradient: for each quadruplet with $\delta>\varepsilon$, with $\text{coeff}=2\delta$,
add $+\text{coeff}$ to the two vector entries of the maximal-sum pair group and
$-\text{coeff}$ to the two entries of the second-maximal group (opposite signs),
then add $2\lambda_{\text{reg}} w$. The data-gradient is zero-sum per quadruplet.
(`src/tropical_gradient_descent.py`)

> **Implementation note.** The vector penalty $\sum_r w_r^2$ equals one half of
> $\sum_{i,j}\Omega_{ij}^2$, so the matrix penalty differs only by a constant
> absorbed into \(\lambda_{\text{reg}}\). The loss and gradient are mutually
> consistent.

## 5. Optimizer (Hybrid)

Define the tropical gradient spread

$$
T(g)=\max(g)-\min(g).
$$

The normalized update is

$$
w \leftarrow w-\eta g,
\qquad
\eta=\frac{\gamma}{T(g)+\varepsilon}.
$$

Stop when $T(g)\le\varepsilon$. After each step, project
$X=D+\Omega$ back onto $\mathcal{D}_n$ (symmetrize, clip negatives to keep
$X\ge0$, zero the diagonal) and recover $\omega=X-D$. The **best iterate** (lowest
loss, including the initial $w=0$) is returned, so $X$ is never worse than $D$.

> **Implementation note.** The build spec wrote
> $\eta=\gamma T(g)/(\sqrt{p}+\varepsilon)$,
> with the tropical norm in the **numerator**; that makes the step grow with the
> gradient and **diverges** (verified: it overflows to NaN), contradicting the
> spec's own "loss must not increase" requirement. Because the data-gradient is
> zero-sum, its tropical norm is a faithful magnitude, so we **normalize by** it
> (denominator) — a bounded, scale-invariant step that still uses the tropical
> norm (never the Euclidean norm), as guardrail #1 requires.

## 6. Relative improvement

$$
\text{relative\_improvement} = \frac{L_2(D)-L_2(X)}{L_2(D)+\varepsilon},
$$

computed on the same quadruplet sample for $D$ and $X$. (`correct_distance_matrix`)

## 7. Neighbor-Joining

With $m$ active nodes, $r_i=\sum_k M_{ik}$ and
$Q_{ij}=(m-2)M_{ij}-r_i-r_j$; merge $\arg\min_{i<j}Q_{ij}$ into a new node $u$ with

$$
\ell_a = \tfrac12 M_{ab} + \frac{r_a-r_b}{2(m-2)},\quad \ell_b = M_{ab}-\ell_a,\quad
M_{uk} = \tfrac12(M_{ak}+M_{bk}-M_{ab}).
$$

Two remaining nodes are joined by an edge of length $M_{ab}$; tiny inputs
($n\le2$) are special-cased and negative display branch lengths are clamped to 0.
The tree is exported as Newick, an edge list, and DOT. (`src/phylogeny.py`)

## Scientific boundary

Reducing tropical four-point violations improves mathematical compatibility with
an additive tree metric. It does **not** establish that the inferred topology is
the biologically correct evolutionary history.
