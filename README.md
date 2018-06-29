# BA_Laura: Benchmarking for Fair Machine Learning Algorithms

This repository provides a framework for benchmarking fair ranking algorithms. It is possible to benchmark fair learning to rank algorithms (model regularization) as well as fair optimization algorithms (post-processing). We implemented three post-processing algorithms alongside three baseline algorithms.

## Datasets

The following datasets are included:

| Code  | Description |
| ----- | ----------- |
| compas | *Correctional Offender Management Profiling for Alternative Sanctions* ([COMPAS](https://github.com/propublica/compas-analysis)): a survey used in some US states for alternative sanctions such as parole * Ke Yang and Julia Stoyanovich. Measuring Fairness in Rankend Outputs. CoRR 2017.  |
| germancredit | [German Credit Scores](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)) (SCHUFA) dataset |
| W3C | [W3C]()|


## Fair Ranking Algorithms:

Our framework currently benchmarks the following fair ranking algorithms

### In-processing:

To be implemented

### Post-processing:

FA\*IR * Meike Zehlike, Francesco Bonchi, Carlos Castillo, Sara Hajian, Mohamed Megahed and Ricardo Baeza-Yates. FA\*IR: A Fair Top-k Ranking Algorithm. CIKM 2017. *(Please use branch FA-IR_CIKM_17 for reproducing experiments of this paper)*

Learning Fair Ranking (LFRanking) * Ke Yang and Julia Stoyanovich. Measuring Fairness in Rankend Outputs. CoRR 2017. 

Fairness of Exposure in Rankings (FOEIR) * Ashudeep Singh and Thorsten Joachims. Fairness of Exposure in Rankings. CoRR 2018. 


## Baseline Algorithms:

### Learning to Rank:

ListNet * Z. Cao, T. Qin, T.-Y. Liu, M.-F. Tsai, and H. Li, “Learning to rank: From pairwise approach to listwise approach,” Tech. Rep., April 2007. [Online]. Available: https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/

### Score-ordering:

Color-blind

Feldman et al.

## Measures:

### Utility: 

Mean Average Precision (MAP) * Liu, Tie-Yan and Xu, Jun and Qin, Tao and Xiong, Wenying and Li, Hang. Letor: Benchmark dataset for research on learning to rank for information retrieval. SIGIR 2007. 

Normalized Discounted Cumulative Gain (NDCG) * J\"{a}rvelin, Kalervo and Kek\"{a}l\"{a}inen, Jaana. Cumulated Gain-based Evaluation of IR Techniques. ACM Trans. Inf. Syst. 2002 

### Fairness:

Normalized discounted KL-divergence (rKL) * Ke Yang and Julia Stoyanovich. Measuring Fairness in Rankend Outputs. CoRR 2017. 

Disparate Impact Ratio (DIR) * Ashudeep Singh and Thorsten Joachims. Fairness of Exposure in Rankings. CoRR 2018. 

Disparate Treatment Ratio (DTR) * Ashudeep Singh and Thorsten Joachims. Fairness of Exposure in Rankings. CoRR 2018. 

Fairness@k based upon * Meike Zehlike, Francesco Bonchi, Carlos Castillo, Sara Hajian, Mohamed Megahed and Ricardo Baeza-Yates. FA\*IR: A Fair Top-k Ranking Algorithm. CIKM 2017. *(Please use branch FA-IR_CIKM_17 for reproducing experiments of this paper)*

### Overall Evaluation:

Normalized Winning Number * Niek Tax and Sander Bockting and Djoerd Hiemstra. A cross-benchmark comparison of 87 learning to rank methods. Information processing \& management 2015. 

## Set-up

### Dependencies

This program was developed and tested in [Python 3.5](https://www.python.org/downloads/release/python-350/). It needs the following packages:

* birkhoff 0.0.5 
* chainer 1.16.0
* CVXOPT 1.2.0
* matplotlib 2.0.0
* numba 0.38.0
* numpy 1.12.0
* pandas 0.19.2
* pip 9.0.1
* scipy 0.18.1

### Environment

For evaluation, we ran the program on a Ubuntu system with 8 kernels and 20 GB ram. We cannot guarantee that a system with less ram will be able to cope with the computational intensive evaluations when set to default values, i.e. k = 40. 

However, while developing the benchmarking suit, we used a machine running Windows 10 with 2 kernels and only 4 GB ram which also could cope with k = 40. Note that it is only possible to set k = 40, 100, 1000, or 1500 due to the alpha_c values available for evaluation of Fairness@k.

## Installation and Starting the Benchmarking Process

1. Clone this repository:
`git clone https://github.com/MilkaLichtblau/BA_Laura.git`

2. In a command-line shell, navigate to the directory where you cloned the repository:
`$ cd ~/BA_Laura`

3. Start the benchmarking process with:
`$ python runBenchmarking.py`

### Preprocessing

We also provide preprocessing for raw data sets to make them available in our framework as well as divide them into five folds in `src/csvProcessing/csvRawDataProcessing.py`. Please have a look at the specifications in the code described in that file for further information.

## Evaluation and Interpretation

NDCG@1, NDCG@5, NDCG@10, MAP, Fairness@k and NWN have values in [0,1] where 1 is the best value.
Performance with regard to DIR and DTR is best at 1. In the diagram we show the diviations from 1, hence we show DIR-1 as well as DTR-1.
rKL's best value is 0 while it also lies in the interval [0,1].

## License
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Contact

Laura Mons

mons at tu-berlin.de
