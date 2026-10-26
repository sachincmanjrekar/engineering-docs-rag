# Evaluation results

18 questions on `PT-200_Pressure_Transmitter_Manual.pdf`, top-k = 4, embedding model `all-MiniLM-L6-v2`, distance threshold 0.75.

| chunk_size | chunks | hit@1 | hit@4 | MRR | abstain |
|---|---|---|---|---|---|
| 256 | 24 | 0.94 | 1.00 | 0.95 | 0.00 |
| 512 | 12 | 0.81 | 1.00 | 0.88 | 0.00 |
| 1024 | 6 | 0.81 | 1.00 | 0.86 | 0.00 |
