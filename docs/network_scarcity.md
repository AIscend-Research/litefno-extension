# Does scarcity travel on a network the operator cannot see? (ext24, H5)

`scripts/network_scarcity.py`, `src/litefno/networks.py`,
`src/litefno/models/graphfno.py`

[ext22](fair_allocation.md) and [ext23](strategic_allocation.md) put a decision
layer on top of the reconstructed ecosystem state and asked what the surrogate's
error costs it. Both assumed the regions being allocated to are independent once
you know the field. They are not, in any system where the regions trade: a
shortage in one place is a shortage in its trading partners a few steps later,
along edges that are nowhere in the PDE.

So this extension borrows the standard epidemiological model of that — an SIS
cascade on a contact graph — couples it to the ecosystem, and adds a
graph-convolutional head to LiteFNO to see whether the network structure is
worth modelling explicitly.

**H5: a graph-convolutional head improves region-level scarcity prediction in
proportion to the share of the trade network that is non-lattice, and by nothing
at all when the network is a pure spatial contact lattice.**

The second half is wrong, and the reason it is wrong is the most useful thing
here. The first half holds.

## The identity that decides what can be claimed

On a periodic `N x N` grid the 4-neighbour lattice Laplacian is circulant, so
its eigenvectors are exactly the 2-D Fourier modes, with eigenvalues

```
lambda(k_y, k_x) = 4 - 2 cos(2 pi k_y / N) - 2 cos(2 pi k_x / N)
```

Measured, over every mode of an 8x8 periodic lattice:

```
max |L e_k - lambda_k e_k|  =  1.52e-14
```

A spectral convolution is therefore *already* a spectral graph convolution on
the pixel lattice — this is the observation the whole graph-network literature
starts from (Bruna et al. 2014; Defferrard et al. 2016), read backwards. A
Fourier layer with free per-mode weights already spans every filter on that
graph. **The only thing a graph layer can add is edges the lattice does not
have.** That is why the independent variable in this extension is the *shortcut
fraction* — the share of edges that are not lattice edges — and not "does a
graph layer help".

## The dynamics, and the closed form they are scored against

Regions are the same 4x4 blocks as ext22/ext23, in the same row-major order
(pinned by a test against `region_gains`, because every graph metric is
permutation-covariant and nothing else in the module could detect a scrambled
graph). Region `r`'s *demand pressure* is `relu(g_r / (share_r * sum g) - 1)`:
scale-free, rectified, zero when everyone gets their share. Scarcity then runs
the discrete NIMFA mean-field SIS update (Van Mieghem, Omic & Kooij 2009):

```
x_{t+1} = x_t + beta (1 - x_t) (A x_t) - gamma x_t + kappa s_t,  clipped to [0,1]
```

That has a known critical point: the cascade dies out iff `beta/gamma <
1/lambda_1(A)` (Wang et al. 2003). Simulating to extinction and bisecting for
the die-out threshold reproduces it on four graph families:

```
               family   lambda_1   1/lambda_1   measured    rel err  shortcuts
        lattice (p=0)     4.0000      0.25000    0.25025   1.00e-03      0.000
  small world (p=0.3)     4.2488      0.23536    0.23579   1.81e-03      0.250
  trade network (p=1)     4.5822      0.21824    0.21880   2.59e-03      0.781
  scale free (BA m=2)     4.5094      0.22176    0.22248   3.24e-03      0.714
```

Worst case 0.3%. The dynamics are what they claim to be, so a model that fails
on them is failing at prediction and not at a broken simulator.

Every graph is run at `beta = 0.8 * gamma / lambda_1`, i.e. at 80% of *its own*
threshold. Without that, sweeping topology would secretly sweep how supercritical
the dynamics are, and the target would saturate at 1 on the well-connected
graphs. Rewiring preserves the edge count exactly, so the arms differ in
topology and in nothing else — and `lambda_1` still moves from 4.00 to 4.58,
which is the whole point: whatever a graph layer is for, it is not for counting
edges.

## 1. The arms

Four models, **4913 parameters each**, same trunk, same seed, same
initialisation, same split, same inputs. They differ in one matrix — the
propagator `Ahat = D^-1/2 (A + I) D^-1/2` inside a `sum_k Ahat^k H W_k`
polynomial filter (Kipf & Welling 2017):

- `identity` — `Ahat = I`. The no-graph ablation, at identical parameter count:
  all powers collapse and the `K+1` weight matrices sum to one effective matrix.
- `lattice` — the 4-neighbour spatial graph. The control for "any graph".
- `rewired` — the true graph, degree-preservingly rewired. The control for
  "right degrees, wrong topology".
- `true` — the network the cascade actually ran on.

Each predicts region scarcity `x_{t+4}` from the field at `t` plus the current
`x_t` painted into a third channel, so no arm has an information advantage.
Held-out region VRMSE, 3 seeds:

```
  p = 0   (shortcut fraction 0.000, lambda_1 4.000)
           arm              VRMSE            vs no-graph
      identity     0.2194 +- 0.0098            0.00%
       lattice     0.2030 +- 0.0290            7.29%
       rewired     0.2222 +- 0.0230           -1.46%
          true     0.2030 +- 0.0290            7.29%
  p = 0.5   (shortcut fraction 0.406, lambda_1 4.312)
      identity     0.2707 +- 0.0187            0.00%
       lattice     0.2878 +- 0.0278           -7.52%
       rewired     0.2475 +- 0.0309            7.45%
          true     0.2171 +- 0.0346           18.58%
  p = 1   (shortcut fraction 0.781, lambda_1 4.445)
      identity     0.2929 +- 0.0041            0.00%
       lattice     0.2970 +- 0.0316           -1.55%
       rewired     0.2638 +- 0.0305            9.78%
          true     0.2187 +- 0.0353           25.15%
```

At `p = 0` the `true` arm *is* the lattice arm, by construction, and it scores
identically to the digit — a sanity check on the plumbing that had to pass.

## 2. H5, against the share of non-lattice edges

The column that tests H5 is `true vs lattice`: the part of the gain that a
convolution provably could not have supplied, since the lattice arm's filter is
one the Fourier trunk already spans.

```
     p   shortcuts       true vs none    true vs lattice    true vs rewired
  0.00       0.000        7.29% +-13.62        0.00% +-0.00         9.02% +-3.87
  0.25       0.188       15.99% +-12.82       21.69% +-7.83        13.27% +-5.00
  0.50       0.406       18.58% +-18.01       24.97% +-6.32        12.65% +-4.24
  0.75       0.656       26.71% +-10.66       28.47% +-7.15        23.24% +-4.32
  1.00       0.781       25.15% +-13.05       26.60% +-7.24        17.51% +-4.56
```

**The topological half of H5 holds.** The gain over any-graph is exactly zero at
zero shortcuts (it must be — same matrix), rises to 21.7% as soon as a fifth of
the edges leave the lattice, and saturates around 27%. It also survives the
harder control: against a graph with the *same degree sequence* and the wrong
wiring, the true network still wins by 12–23%, so the layer is using topology
and not just node degree.

**The null half of H5 is refuted.** At `p = 0` the lattice arm beats the no-graph
arm by 7.3% — on a graph whose every filter the spectral trunk already spans.
Representable is not the same as learned. The identity arm has the capacity to
express region coupling only by routing it through the trunk and the pooling
step; the graph arm has it as an architectural prior, and a prior worth 7% is
still worth having even when it adds no expressive power. The honest reading:
*hard-wiring the lattice buys optimisation, hard-wiring the true network buys
optimisation plus 20-27% of genuinely new capacity*, and only the second scales
with topology.

## 3. Where to put the sentinels

The contact-tracing question, on the same graphs: with 3 monitored regions, how
early is a cascade caught, and does centrality beat chance?

```
                 graph         rule      delay   spread at detect   missed   tie
         lattice (p=0)  eigenvector       2.19             0.2891        0   yes
         lattice (p=0)       degree       3.16             0.4219        0   yes
         lattice (p=0)       random       2.32             0.3119        0   yes
   trade network (p=1)  eigenvector       2.16             0.2754        0    no
   trade network (p=1)       degree       2.16             0.2754        0    no
   trade network (p=1)       random       2.38             0.3074        0   yes
```

`tie` marks a rule whose score is *constant across regions* — the lattice is
regular, so degree centrality is uniform and eigenvector centrality is uniform
(Perron-Frobenius on a regular graph gives the flat vector), and the "rule"
is an arbitrary tie-break. Those rows are not evidence about the rule, and they
are labelled rather than quietly reported; that is why the degree row on the
lattice looks worst.

On the trade network, where the scores are not degenerate, centrality-placed
sentinels catch the cascade at 27.5% of the network infected versus 30.7% for
random — a real but small edge, worth about 0.22 steps of warning. Both
centrality rules pick the same set. **The placement result is much weaker than
the modelling result**, and at 16 nodes with 3 sentinels there is not much room
between "best 3" and "random 3" for it to be otherwise.

## What this rules out, and what it does not

Ruled out: that adding a graph layer to a Fourier operator is justified by
"capturing network effects" in general. On a spatial lattice it is not — the
operator already contains that graph convolution exactly, and the residual is
1.5e-14. Any paper claiming a graph layer adds *expressive* capacity over a
spectral operator on a grid owes a demonstration that its edges are not lattice
edges.

Ruled out: that the effect measured here is a degree effect. The
degree-preserving rewired control keeps every node's degree and still loses by
12–23%.

Not ruled out: that a better-optimised identity arm closes the 7.3% lattice gap.
The claim here is about what these architectures learn under a fixed budget, not
about the closure of their hypothesis classes — which is precisely the
distinction the closed form lets us make.

Not ruled out: that the trade network could be *learned* rather than supplied.
Every arm here is given its graph. A layer that infers the adjacency from data
is the obvious next question and is not attempted.

## Honest limits

- **The network is synthetic and the coupling is stipulated.** Nothing in
  Gray-Scott or in the oscillatory testbed says which regions trade. The
  Watts-Strogatz family is a modelling choice; the demand pressure that drives
  the cascade is a modelling choice; only the SIS dynamics and the threshold are
  pinned to closed forms.
- **16 nodes.** The graph is 4x4 blocks, so the spectral gaps between families
  are small, the sentinel experiment has almost no room, and the scale-free arm
  is barely scale-free. Nothing here should be read as a claim about large
  networks.
- **The head is unconstrained.** Scarcity lives in [0,1] and the model is not
  told so — an early sigmoid head saturated at zero and made every arm score
  worse than the mean predictor, so the constraint was dropped rather than
  faked. The violation is measured instead: the worst excursion outside [0,1] is
  0.021 and the mean is 0.006-0.012, larger for the graph arms than for the
  identity arm. The arms are compared on the same unconstrained footing.
- **Subcritical only.** Every run sits at 80% of its own epidemic threshold. Near
  or above threshold the target saturates and the comparison changes character;
  that regime is not tested.
- **Three seeds.** The per-arm error bars overlap at every `p`; the paired
  columns are what carry the result, because the arms share initialisation and
  data seed-by-seed. The `+-` figures are standard deviations across 3 paired
  differences, not confidence intervals.

## Reproducing

```bash
python3 scripts/network_scarcity.py           # ~25 minutes on CPU (60 trainings)
python3 scripts/network_scarcity.py --quick   # plumbing check only
```

Outputs: `results/extensions/ext24_threshold.csv`, `ext24_arms.csv`,
`ext24_sweep.csv`, `ext24_sentinels.csv`, and
`figures/extensions/ext24_network_scarcity.png`.

## References

- Bruna, Zaremba, Szlam & LeCun (2014), *Spectral Networks and Locally Connected
  Networks on Graphs*.
- Defferrard, Bresson & Vandergheynst (2016), *Convolutional Neural Networks on
  Graphs with Fast Localized Spectral Filtering*.
- Kipf & Welling (2017), *Semi-Supervised Classification with Graph Convolutional
  Networks*.
- Van Mieghem, Omic & Kooij (2009), *Virus Spread in Networks*.
- Wang, Chakrabarti, Wang & Faloutsos (2003), *Epidemic Spreading in Real
  Networks: An Eigenvalue Viewpoint*.
- Watts & Strogatz (1998), *Collective Dynamics of 'Small-World' Networks*.
