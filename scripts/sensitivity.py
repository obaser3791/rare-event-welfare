"""Sensitivity of the transfusion targeting-regret entry (Table 5) to the
calibration slopes: age slope in {0.10,0.18,0.30}, comorbidity log-odds in
{0.50,0.90,1.30}."""
import json
from empirical import empirical_table
res = []
for ba in (0.10, 0.18, 0.30):
    for bc in (0.50, 0.90, 1.30):
        tab, _ = empirical_table(beta_age=ba, beta_com=bc)
        r = tab[0]
        res.append(dict(beta_age=ba, beta_com=bc, outcome=r["outcome"], rel_regret_pct=r["rel_regret_pct"]))
        print(f"age {ba:.2f} com {bc:.2f}  {r['outcome']}: {r['rel_regret_pct']:.2f}%")
json.dump(res, open("sensitivity_results.json", "w"), indent=1)
