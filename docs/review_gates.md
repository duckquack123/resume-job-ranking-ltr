# Review Gates (Self-Audit Checklist)

Before submitting a review packet, ensure every box is checked:
- [ ] Are results based on enough samples? Report n and a bootstrap CI; don't draw conclusions from 5 jobs.
- [ ] Could any label or feature leak into the other (circularity)?
- [ ] Are splits disjoint for jobs, companies and CVs? Are the tests independent of the dedup step (different seed, shingle unit, threshold)?
- [ ] Did any diagnostic touch gold or dev jobs?
- [ ] Is any claim about a cause tested or only asserted?
- [ ] Does every number trace to a raw log line I can show?
- [ ] Did I apply every requested change? Compare the file before and after and list each change. If a file is unchanged, say so.
- [ ] Did I remove unverified claims (corpus-wide vs subset counts, sample vs corpus statistics)?
