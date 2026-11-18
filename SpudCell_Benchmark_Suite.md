# SpudCell Synthetic Cell Computational Benchmark & Stabilization Suite

**Overview:** This open-science benchmark models population survival, ribosome dilution kinetics, minimal-parasite genome drift, and active plasmid segregation across seven computational regimes for the 7-plasmid SpudCell synthetic cell system (Generations 0 to 8).

---

## 1. Seven-Regime Comparison Summary (Generations 6 & 8)

| Simulation Regime | Gen 6 Active Cells | Gen 8 Active Cells | Gen 8 Full 7-Plasmid (%) | Gen 8 Avg Ribosomes / Cell |
| :--- | :--- | :--- | :--- | :--- |
| **1. Baseline (No External Ribosomes)** | 0 *(Extinct)* | 0 *(Extinct)* | 0.0% | 0.0 |
| **2. External Ribosomes (Unselected, Rate=18)** | 1,000 | 1,000 | 2.5% | 32.2 |
| **3. Fusion Penalty (Lambda=0.8, Rate=18)** | 1,000 | 1,000 | 3.3% | 14.4 |
| **4. Low Ribosome Payload (Lambda=2.5, Rate=5)** | 1,000 | 1,000 | 2.7% | 10.9 |
| **5. Strict 7-Plasmid Selection (2 Copies)** | 156 | 201 | 100.0% | 34.5 |
| **6. High Copy Number (5 Copies + Strict)** | 1,000 | 1,000 | 100.0% | 34.9 |
| **7. Active ParABS Partitioning (90% + Strict)** | 1,000 | 1,000 | 100.0% | 34.7 |

---

## 2. Key Engineering Findings for Synthetic Cell Researchers

1. **Translational Arrest Rescue:** Without external ribosome replenishment, cells fall below the 5-ribosome viability threshold and go extinct by Generation 6. Continuous liposome delivery stabilizes intracellular counts at **10.9–34.9 ribosomes/cell**, even under a **68% fusion penalty** (`lambda = 0.8`) or **low payload** (`rate = 5`).
2. **Quantifying "Minimal-Parasite" Drift:** Supplying external ribosomes without 7-plasmid selection allows minimal cells carrying only the fusion (`plasmid[0]`) and Phi29 replication (`plasmid[1]`) plasmids to out-compete complete cells, eroding full-genome retention down to **2.5%** by Generation 8.
3. **Resolving the Selection Mortality Bottleneck:** Enforcing strict 7-plasmid selection preserves **100.0%** genome integrity but slows population growth to **201 cells** at Generation 8 due to binomial plasmid loss. Raising initial copy numbers to **5 copies** or adding a **90% efficient `ParABS` active partitioning module** restores exponential growth to the **1,000-cell** bioreactor cap by Generation 4.

---

## 3. Reproducible Python Simulation Suite

```python
import numpy as np
import pandas as pd

def run_spudcell_suite(generations=8, init_cells=100, init_copies=2,
                       external_ribosome_rate=0, fusion_lambda=2.5,
                       require_all_7=False, parabs_eff=0.0, seed=42):
    np.random.seed(seed)
    cells = [{'plasmids': [init_copies]*7, 'ribosomes': 120, 'active': True} for _ in range(init_cells)]
    history = []

    for gen in range(generations + 1):
        active = [c for c in cells if c['active']]
        full = [c for c in active if all(p >= 1 for p in c['plasmids'])]
        history.append({
            'Gen': gen,
            'Active': len(active),
            'Full7': len(full),
            'PctFull': round((len(full)/len(active)*100) if active else 0, 1),
            'AvgRibo': round(np.mean([c['ribosomes'] for c in active]) if active else 0, 1)
        })
        if gen == generations or not active:
            break

        next_gen = []
        for c in active:
            if c['plasmids'][0] >= 1:
                c['ribosomes'] += np.random.poisson(fusion_lambda) * external_ribosome_rate
            c['ribosomes'] = int(c['ribosomes'] * 0.88)

            if c['ribosomes'] < 5 or c['plasmids'][1] < 1:
                continue
            if require_all_7 and not all(p >= 1 for p in c['plasmids']):
                continue

            rep = [p * 2 for p in c['plasmids']]
            d1_p, d2_p = [], []
            for p_cnt in rep:
                if parabs_eff > 0 and p_cnt >= 2 and np.random.rand() < parabs_eff:
                    rem = p_cnt - 2
                    b1 = np.random.binomial(rem, 0.5)
                    d1_p.append(1 + b1)
                    d2_p.append(1 + (rem - b1))
                else:
                    c1 = np.random.binomial(p_cnt, 0.5)
                    d1_p.append(c1)
                    d2_p.append(p_cnt - c1)

            d1_r = np.random.binomial(c['ribosomes'], 0.5)
            d2_r = c['ribosomes'] - d1_r

            if d1_r >= 5 and d1_p[1] >= 1 and (not require_all_7 or all(p >= 1 for p in d1_p)):
                next_gen.append({'plasmids': d1_p, 'ribosomes': d1_r, 'active': True})
            if d2_r >= 5 and d2_p[1] >= 1 and (not require_all_7 or all(p >= 1 for p in d2_p)):
                next_gen.append({'plasmids': d2_p, 'ribosomes': d2_r, 'active': True})

        if len(next_gen) > 1000:
            idx = np.random.choice(len(next_gen), 1000, replace=False)
            next_gen = [next_gen[i] for i in idx]
        cells = next_gen

    return pd.DataFrame(history)

if __name__ == "__main__":
    regimes = {
        "Baseline": run_spudcell_suite(external_ribosome_rate=0),
        "Ext_Ribo": run_spudcell_suite(external_ribosome_rate=18),
        "Fusion_Penalty": run_spudcell_suite(external_ribosome_rate=18, fusion_lambda=0.8),
        "Low_Payload": run_spudcell_suite(external_ribosome_rate=5, fusion_lambda=2.5),
        "Strict_7": run_spudcell_suite(external_ribosome_rate=18, require_all_7=True),
        "High_Copy_5": run_spudcell_suite(external_ribosome_rate=18, init_copies=5, require_all_7=True),
        "ParABS_90": run_spudcell_suite(external_ribosome_rate=18, init_copies=2, require_all_7=True, parabs_eff=0.90),
    }
    for name, df in regimes.items():
        print(f"=== {name} ===")
        print(df.to_string(index=False))
```
