Using these datasets was inspired by the following paper:
Ke Yang and Julia Stoyanovich. "Measuring Fairness in Ranked Outputs." arXiv preprint arXiv:1610.08559 (2016).

We downloaded them from: 
https://github.com/DataResponsibly/FairRank/tree/6c1ab4a4751448443c86921976b725530e0cc84f/datasets

For our experiments we preprocessed the data, extracting only "Recidivism_rawscore" and the protected attribute column ("sex" and "race" respectively). Additionally, we calculated the reverse of the "Recidivism_rawscore" using 1 - "Recidivism_rawscore".