## Dynamic‑Alpha Gradient Optimizer

*A drop‑in replacement for `torch.optim.SGD` / `Adam(W)` with automatic α‑control, per‑layer RMS scaling, and global RMS shrink.*
This repository contains all of the testing code and submodules used for reproducibility of the results discussed in (link soon) along with the PyTorch libarary extension.
---

DAG lives in

```text
optim/sgd.py        # class DAG
```

and is imported exactly like any PyTorch optimizer:

```python
from optim.sgd import DAG
```

---

### Quick‑start example

```python
import torch
from optim.sgd import DAG

model = MyNetwork()
criterion = torch.nn.CrossEntropyLoss()

opt = DAG(
    model.parameters(),
    lr=3e-4,                # base learning‑rate  η
    weight_decay=1e-2,      # optional L2 ( AdamW‑style if fused=False )
    momentum=0.9,           # standard SGD momentum (optional)
    k_val=1.5,              # κ  – per‑layer RMS target
    lambda_rms=0.3,         # λ_rms – when RMS(update)<0.3·RMS0 ⇒ global shrink begins
    s_min=0.1,              # floor for global shrink s_t
    hyper=dict(             # α‑controller knobs (rarely need tuning)
        tau=1.5,
        p_star=0.10,
        beta=1/3,
        eta=0.3,
        rho=0.1
    ),
    shrink=dict(            # RMS‑shrink knobs
        ema_beta=0.98,
        warmup_steps=500,
        gamma=1.0
    )
)

for epoch in range(E):
    for x, y in loader:
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
```

---

### Full constructor signature

```python
DAG(
    params,                         # iterable of Tensors / dicts
    lr                = 1e-3,       # η       base LR
    momentum          = 0.0,
    dampening         = 0.0,
    weight_decay      = 0.0,        # λ_wd   coupled L2 (set 0 for AdamW‑style decoupling)
    k_val             = 2.0,        # κ      target per‑layer RMS
    k_sched           = None,       # callable(step)->k  (cosine helper included)
    nesterov          = False,
    maximize          = False,      # ascent instead of descent
    foreach           = None,       # PyTorch foreach kernels
    differentiable    = False,
    fused             = None,
    # α‑controller
    hyper = dict(
        tau      = 1.25,            # tanh active threshold
        p_star   = 0.10,            # target sat. fraction
        kappa    = None,            # (auto‑derived if None)
        beta     = 1/3,             # dim factor exponent
        eta      = 0.3,             # sat feedback
        rho      = 0.1,             # α EMA
        eps      = 1e-5,
        alpha_min= 1e-12,
        alpha_max= 1e12,
    ),
    # RMS‑shrink
    shrink = dict(
        lambda_rms  = 0.3,          # λ_rms
        s_min       = 0.1,          # min global shrink
        gamma       = 1.0,          # shrink curve exponent
        ema_beta    = 0.98,         # RMS(update) EMA β
        warmup_steps= 500,          # steps to lock s_t=1
    ),
    # statistics toggles
    use_exact_sigma = False,        # use Apex MT-std if available
    sigma_every     = 1,            # compute exact σ every N steps
    sat_every       = 10,           # compute saturation every N steps
)
```


---

### Tips & remarks

* **Momentum works.**  When `momentum>0`, DAG behaves like $\text{SGD}_{\text{mom}}$ with adaptive scalars.
* **Coupled vs. decoupled weight decay.**  Set `weight_decay=λ` to couple with LR (SGD‑style).
  To mimic AdamW, call `torch.nn.utils.weight_norm` or implement decay manually.
* **Sparse layers.**  For massive embedding tables combine DAG for dense layers with sparse‑Adam for tables.
* **Debugging.**

  * Print `opt.s_t`, `opt.k_val`, `layer_state['alpha']` to watch the dials.
  * Typical ranges during stable training: `0.1 < s_t ≤ 1`, `0.5 < m_ℓκ ≤ 2`, `|α\,ĝ|_∞ ≈ 1–2`.

---

Happy training — and may your updates always stay in the Goldilocks zone 🚀
