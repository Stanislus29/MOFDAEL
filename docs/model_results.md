**Transmission Line Fault Detection: Model Comparison**  
Dataset: classdetectfault_data.csv, shape (7861, 11), 0 rows removed.  
   
 Features: Ia, Ib, Ic, Va, Vb, Vc. Targets: G, C, B, A (fault type indicators).  
   
 Train/test split: 6288 / 1573 samples (LSTM used event-based split: 2443 / 611 events).  
**Class Distribution**  
| | | | | |  
|-|-|-|-|-|  
| **Target** | **Class 0** | **Class 0 %** | **Class 1** | **Class 1 %** |   
| G | 4465 | 56.80% | 3396 | 43.20% |   
| C | 4628 | 58.87% | 3233 | 41.13% |   
| B | 3494 | 44.45% | 4367 | 55.55% |   
| A | 3369 | 42.86% | 4492 | 57.14% |   
   
**Per-Target Results by Model**  
**G**  
| | | | | |  
|-|-|-|-|-|  
| **Model** | **Accuracy** | **Precision** | **Recall** | **F1** |   
| SVM (RBF) | 0.8856 | 0.8105 | 0.9653 | 0.8811 |   
| Decision Tree | 0.8773 | 0.7964 | 0.9682 | 0.8739 |   
| K-NN | 0.8970 | 0.8930 | 0.8698 | 0.8812 |   
| ANN | 0.8741 | 0.8834 | 0.8220 | 0.8516 |   
| RNN | 0.8729 | 0.8670 | 0.8394 | 0.8529 |   
| CNN | 0.8824 | 0.7942 | 0.9884 | 0.8807 |   
| LSTM | 0.8304 | 0.7265 | 0.9649 | 0.8289 |   
   
**C**  
| | | | | |  
|-|-|-|-|-|  
| **Model** | **Accuracy** | **Precision** | **Recall** | **F1** |   
| SVM (RBF) | 0.9930 | 0.9967 | 0.9853 | 0.9909 |   
| Decision Tree | 0.9949 | 0.9967 | 0.9902 | 0.9934 |   
| K-NN | 0.9975 | 0.9967 | 0.9967 | 0.9967 |   
| ANN | 0.9917 | 0.9983 | 0.9804 | 0.9893 |   
| RNN | 0.9898 | 0.9983 | 0.9755 | 0.9868 |   
| CNN | 0.9968 | 0.9984 | 0.9935 | 0.9959 |   
| LSTM | 0.9785 | 1.0000 | 0.9505 | 0.9746 |   
   
**B**  
| | | | | |  
|-|-|-|-|-|  
| **Model** | **Accuracy** | **Precision** | **Recall** | **F1** |   
| SVM (RBF) | 0.9917 | 0.9976 | 0.9870 | 0.9923 |   
| Decision Tree | 0.9898 | 0.9976 | 0.9835 | 0.9905 |   
| K-NN | 0.9962 | 0.9988 | 0.9941 | 0.9965 |   
| ANN | 0.9905 | 1.0000 | 0.9823 | 0.9911 |   
| RNN | 0.9841 | 1.0000 | 0.9706 | 0.9851 |   
| CNN | 0.9968 | 1.0000 | 0.9941 | 0.9970 |   
| LSTM | 0.9952 | 1.0000 | 0.9919 | 0.9960 |   
   
**A**  
| | | | | |  
|-|-|-|-|-|  
| **Model** | **Accuracy** | **Precision** | **Recall** | **F1** |   
| SVM (RBF) | 0.9981 | 0.9978 | 0.9989 | 0.9983 |   
| Decision Tree | 0.9962 | 1.0000 | 0.9933 | 0.9966 |   
| K-NN | 1.0000 | 1.0000 | 1.0000 | 1.0000 |   
| ANN | 0.9962 | 1.0000 | 0.9933 | 0.9966 |   
| RNN | 0.9943 | 0.9955 | 0.9944 | 0.9949 |   
| CNN | 0.9994 | 1.0000 | 0.9989 | 0.9994 |   
| LSTM | 0.9922 | 1.0000 | 0.9869 | 0.9934 |   
   
**Overall Model Summary**  
| | | | | | | |  
|-|-|-|-|-|-|-|  
| **Model** | **Mean Accuracy** | **Mean Precision** | **Mean F1** | **Mean Latency (ms)** | **Mean Robustness** | **Overall Fault-Occurred Accuracy** |   
| SVM (RBF) | 0.9671 | 0.9506 | 0.9657 | 1.4676 | 0.9661 | 0.9981 |   
| Decision Tree | 0.9646 | 0.9477 | 0.9636 | 1.0060 | 0.9413 | 0.9981 |   
| K-NN | 0.9727 | 0.9721 | 0.9686 | 20.4359 | 0.9540 | 1.0000 |   
| ANN | 0.9631 | 0.9704 | 0.9571 | 0.9652 | 0.9635 | 0.9975 |   
| RNN | 0.9603 | 0.9652 | 0.9549 | 1.0631 | 0.9601 | 0.9949 |   
| CNN | 0.9688 | 0.9481 | 0.9683 | 83.9715 | 0.9685 | 0.9994 |   
| LSTM | 0.9491 | 0.9316 | 0.9482 | 90.2303 | 0.9156 | 0.9994 |   
   
Latency is measured per row/sample for SVM, Decision Tree, K-NN, ANN, RNN, and CNN (inference), and per sample (streaming) for LSTM. Robustness is the mean accuracy/F1 retained across the noise sweep (1%, 5%, 10%, 20%).  
**Robustness Sweep by Model, All Targets**  
Mean accuracy ± standard deviation across 10 trials per noise level. G is the hardest target across models and shows the widest spread; A and B are comparatively stable up to 10% noise for most models.  
**G**  
| | | | | |  
|-|-|-|-|-|  
| **Model** | **Noise 1%** | **Noise 5%** | **Noise 10%** | **Noise 20%** |   
| SVM (RBF) | 0.8842 ± 0.0009 | 0.8785 ± 0.0028 | 0.8615 ± 0.0039 | 0.8187 ± 0.0053 |   
| Decision Tree | 0.8467 ± 0.0028 | 0.7709 ± 0.0081 | 0.7070 ± 0.0120 | 0.6172 ± 0.0126 |   
| K-NN | 0.8226 ± 0.0040 | 0.8530 ± 0.0047 | 0.8576 ± 0.0041 | 0.8446 ± 0.0049 |   
| ANN | 0.8756 ± 0.0015 | 0.8713 ± 0.0049 | 0.8655 ± 0.0056 | 0.8407 ± 0.0084 |   
| RNN | 0.8736 ± 0.0042 | 0.8739 ± 0.0063 | 0.8679 ± 0.0082 | 0.8397 ± 0.0108 |   
| CNN | 0.8821 ± 0.0007 | 0.8752 ± 0.0032 | 0.8548 ± 0.0050 | 0.8036 ± 0.0065 |   
| LSTM | 0.7663 ± 0.0012 | 0.7670 ± 0.0025 | 0.7671 ± 0.0041 | 0.7658 ± 0.0071 |   
   
**C**  
| | | | | |  
|-|-|-|-|-|  
| **Model** | **Noise 1%** | **Noise 5%** | **Noise 10%** | **Noise 20%** |   
| SVM (RBF) | 0.9915 ± 0.0008 | 0.9797 ± 0.0029 | 0.9329 ± 0.0079 | 0.8631 ± 0.0066 |   
| Decision Tree | 0.9837 ± 0.0020 | 0.8926 ± 0.0050 | 0.7833 ± 0.0075 | 0.6499 ± 0.0086 |   
| K-NN | 0.9972 ± 0.0006 | 0.9894 ± 0.0019 | 0.9758 ± 0.0027 | 0.9449 ± 0.0047 |   
| ANN | 0.9916 ± 0.0004 | 0.9847 ± 0.0026 | 0.9590 ± 0.0051 | 0.9015 ± 0.0058 |   
| RNN | 0.9879 ± 0.0010 | 0.9606 ± 0.0033 | 0.9027 ± 0.0049 | 0.8097 ± 0.0064 |   
| CNN | 0.9964 ± 0.0008 | 0.9723 ± 0.0032 | 0.9427 ± 0.0037 | 0.8861 ± 0.0058 |   
| LSTM | 0.9788 ± 0.0003 | 0.9773 ± 0.0010 | 0.9650 ± 0.0025 | 0.9430 ± 0.0036 |   
   
**B**  
| | | | | |  
|-|-|-|-|-|  
| **Model** | **Noise 1%** | **Noise 5%** | **Noise 10%** | **Noise 20%** |   
| SVM (RBF) | 0.9911 ± 0.0006 | 0.9835 ± 0.0028 | 0.9403 ± 0.0043 | 0.8617 ± 0.0088 |   
| Decision Tree | 0.9707 ± 0.0024 | 0.8919 ± 0.0085 | 0.8053 ± 0.0059 | 0.6920 ± 0.0088 |   
| K-NN | 0.9962 ± 0.0002 | 0.9925 ± 0.0011 | 0.9823 ± 0.0019 | 0.9561 ± 0.0028 |   
| ANN | 0.9906 ± 0.0008 | 0.9799 ± 0.0028 | 0.9425 ± 0.0049 | 0.8692 ± 0.0047 |   
| RNN | 0.9839 ± 0.0004 | 0.9799 ± 0.0011 | 0.9546 ± 0.0047 | 0.8861 ± 0.0068 |   
| CNN | 0.9964 ± 0.0006 | 0.9723 ± 0.0026 | 0.9310 ± 0.0039 | 0.8576 ± 0.0029 |   
| LSTM | 0.9698 ± 0.0009 | 0.9717 ± 0.0025 | 0.9703 ± 0.0033 | 0.9563 ± 0.0054 |   
   
**A**  
| | | | | |  
|-|-|-|-|-|  
| **Model** | **Noise 1%** | **Noise 5%** | **Noise 10%** | **Noise 20%** |   
| SVM (RBF) | 0.9975 ± 0.0004 | 0.9909 ± 0.0008 | 0.9620 ± 0.0028 | 0.9031 ± 0.0090 |   
| Decision Tree | 0.9640 ± 0.0030 | 0.8897 ± 0.0066 | 0.8316 ± 0.0086 | 0.7401 ± 0.0102 |   
| K-NN | 0.9998 ± 0.0003 | 0.9970 ± 0.0008 | 0.9893 ± 0.0025 | 0.9671 ± 0.0039 |   
| ANN | 0.9961 ± 0.0007 | 0.9901 ± 0.0015 | 0.9761 ± 0.0028 | 0.9380 ± 0.0044 |   
| RNN | 0.9950 ± 0.0006 | 0.9914 ± 0.0013 | 0.9749 ± 0.0029 | 0.9186 ± 0.0048 |   
| CNN | 0.9992 ± 0.0003 | 0.9857 ± 0.0041 | 0.9519 ± 0.0027 | 0.8957 ± 0.0044 |   
| LSTM | 0.9474 ± 0.0015 | 0.9417 ± 0.0015 | 0.9392 ± 0.0019 | 0.9324 ± 0.0046 |   
   
Decision Tree has the steepest and most consistent degradation across all four targets. LSTM is the flattest across noise levels on G, B, but starts from the lowest baseline on G and A. K-NN and CNN hold up best on C, B, A but K-NN is notably weaker than the others on G at low noise, then overtakes at higher noise. ANN degrades more gently than RNN on C and B, but the two track closely on G and A.  
**Accuracy by Fault Combination Multiplicity**  
No single-line fault examples were present in the test set for any model. SVM was not included in this evaluation.  
| | | | | | | |  
|-|-|-|-|-|-|-|  
| **Model** | **Double-line (n)** | **Double-line Acc** | **Triple-line (n)** | **Triple-line Acc** | **Quadruple-line (n)** | **Quadruple-line Acc** |   
| ANN | 443 | 0.9707 | 437 | 0.7986 | 211 | 0.4218 |   
| CNN | 443 | 0.9910 | 437 | 0.5835 | 211 | 0.9621 |   
| Decision Tree | 443 | 0.9729 | 437 | 0.5789 | 211 | 0.8957 |   
| K-NN | 443 | 0.9977 | 437 | 0.8215 | 211 | 0.5687 |   
| LSTM | 449 | 0.9978 | 544 | 0.4871 | 224 | 0.7321 |   
| RNN | 443 | 0.9661 | 437 | 0.7506 | 211 | 0.4645 |   
   
Note: this update supersedes the earlier LSTM combination-accuracy figures, drawn from a rerun alongside ANN and RNN.  
**Accuracy by Specific Fault Combination**  
Only combinations with test examples present are shown. Triple-line faults split sharply by combination: G+B+A stays high across all models, while C+B+A collapses.  
| | | | | | |  
|-|-|-|-|-|-|  
| **Model** | **G+A (double)** | **C+B (double)** | **G+B+A (triple)** | **C+B+A (triple)** | **G+C+B+A (quad)** |   
| ANN | 0.9628 (n=242) | 0.9801 (n=201) | 0.9496 (n=238) | 0.6181 (n=199) | 0.4218 (n=211) |   
| CNN | 0.9959 (n=242) | 0.9851 (n=201) | 0.9790 (n=238) | 0.1106 (n=199) | 0.9621 (n=211) |   
| Decision Tree | 0.9711 (n=242) | 0.9751 (n=201) | 0.9454 (n=238) | 0.1407 (n=199) | 0.8957 (n=211) |   
| K-NN | 0.9959 (n=242) | 1.0000 (n=201) | 0.9748 (n=238) | 0.6382 (n=199) | 0.5687 (n=211) |   
| LSTM | 0.9956 (n=225) | 1.0000 (n=224) | 0.9242 (n=264) | 0.0750 (n=280) | 0.7321 (n=224) |   
| RNN | 0.9711 (n=242) | 0.9602 (n=201) | 0.9160 (n=238) | 0.5528 (n=199) | 0.4645 (n=211) |   
   
ANN and RNN both struggle on quadruple-line faults relative to CNN, Decision Tree, and LSTM, closer to K-NN's weak spot there. Neither breaks the C+B+A collapse other models show, though both hold up better on it than CNN, Decision Tree, and LSTM.  
**Training Time and Model Complexity**  
| | | |  
|-|-|-|  
| **Model** | **Training Time (total, s)** | **Notes** |   
| SVM (RBF) | 8.24 | C=8.0, gamma=scale; support vectors range 401–1828 across targets |   
| Decision Tree | 0.17 | max depth 10; leaf count range 18–94 |   
| K-NN | 0.09 | K=5, distance-weighted, minkowski p=2; 6288 reference points stored |   
| ANN | 26.04 | architecture [64, 32], 30 epochs, batch size 64, CPU |   
| RNN | 34.54 | hidden size 64, 2 layers, 30 epochs, batch size 64, CPU |   
| CNN | 161.24 | filters [32, 64], kernel size 2, batch size 32 |   
| LSTM | 406.05 | units [64, 32] bidirectional, batch size 16 events, event-sequence input |   
   
**Hypothesis Testing (H1–H4)**  
Evaluated on the 339 rows common across all six models (ann, cnn, dtr, knn, lstm, rnn), dropping rows outside that common set. The dropped sample counts were 1,234 rows each for ANN, CNN, DTR, KNN, and RNN, and 1,336 rows for LSTM due to its event-based sequence alignment.  
**H1: Fault Detection Accuracy (overall fault occurred, vs relay baseline)**  
Cochran's Q omnibus test across all seven methods (six models plus relay): Q = 5.1429, df = 6, p = 0.5256. No statistically significant difference was detected among the methods.  
| | | | | | |  
|-|-|-|-|-|-|  
| **Model vs Relay** | **p-value** | **Test** | **Accuracy Diff (pp)** | **95% CI** | **Odds Ratio** |   
| ANN | 1.0000 | exact binomial | -0.29 | [-0.87, 0.28] | 0.0000 |   
| CNN | 1.0000 | exact | 0.00 | [0.00, 0.00] | n/a |   
| Decision Tree | 1.0000 | exact binomial | -0.29 | [-0.87, 0.28] | 0.0000 |   
| K-NN | 1.0000 | exact | 0.00 | [0.00, 0.00] | n/a |   
| LSTM | 1.0000 | exact binomial | -0.29 | [-0.87, 0.28] | 0.0000 |   
| RNN | 0.5000 | exact binomial | -0.59 | [-1.41, 0.23] | 0.0000 |   
   
No model differs significantly from the relay baseline on overall fault detection.  
**H2: Per-Line Classification Performance**  
Cochran's Q omnibus test per target across all six models (df = 5):  
| | | | |  
|-|-|-|-|  
| **Target** | **Q** | **df** | **p-value** |   
| G | 11.2715 | 5 | 0.0463 |   
| C | 17.6190 | 5 | 0.0035 |   
| B | 7.3333 | 5 | 0.1970 |   
| A | 8.0000 | 5 | 0.1562 |   
   
Targets G (p = 0.0463) and C (p = 0.0035) exhibit statistically significant omnibus differences across models, while targets B and A do not. Pairwise McNemar comparisons:  
| | | | | |  
|-|-|-|-|-|  
| **Target** | **Comparison** | **p-value** | **Test** | **Diff (pp)** |   
| G | CNN vs DTR | 0.4545 | exact binomial | 1.18 |   
| G | LSTM vs DTR | 0.4545 | exact binomial | -1.18 |   
| G | CNN vs KNN | 0.2225 | chi2 (cont. corrected) | -2.65 |   
| G | LSTM vs KNN | 0.0196 | chi2 (cont. corrected) | -5.01 |   
| G | CNN vs RNN | 0.4610 | chi2 (cont. corrected) | -1.77 |   
| G | LSTM vs RNN | 0.0933 | chi2 (cont. corrected) | -4.13 |   
| G | CNN vs ANN | 0.3222 | chi2 (cont. corrected) | -2.36 |   
| G | LSTM vs ANN | 0.0608 | chi2 (cont. corrected) | -4.72 |   
| C | CNN vs DTR | 1.0000 | exact binomial | 0.29 |   
| C | LSTM vs DTR | 0.4531 | exact binomial | -0.88 |   
| C | CNN vs KNN | 1.0000 | exact binomial | 0.29 |   
| C | LSTM vs KNN | 0.3750 | exact binomial | -0.88 |   
| C | CNN vs RNN | 0.0156 | exact binomial | 2.06 |   
| C | LSTM vs RNN | 0.3750 | exact binomial | 0.88 |   
| C | CNN vs ANN | 0.0156 | exact binomial | 2.06 |   
| C | LSTM vs ANN | 0.3750 | exact binomial | 0.88 |   
| B | CNN vs DTR | 0.5000 | exact binomial | 0.59 |   
| B | LSTM vs DTR | 0.5000 | exact binomial | 0.59 |   
| B | CNN vs KNN | 1.0000 | exact binomial | 0.29 |   
| B | LSTM vs KNN | 1.0000 | exact binomial | 0.29 |   
| B | CNN vs RNN | 0.2500 | exact binomial | 0.88 |   
| B | LSTM vs RNN | 0.2500 | exact binomial | 0.88 |   
| B | CNN vs ANN | 0.5000 | exact binomial | 0.59 |   
| B | LSTM vs ANN | 0.5000 | exact binomial | 0.59 |   
| A | CNN vs DTR | 1.0000 | exact | 0.00 |   
| A | LSTM vs DTR | 0.5000 | exact binomial | -0.59 |   
| A | CNN vs KNN | 1.0000 | exact | 0.00 |   
| A | LSTM vs KNN | 0.5000 | exact binomial | -0.59 |   
| A | CNN vs RNN | 0.2500 | exact binomial | 0.88 |   
| A | LSTM vs RNN | 1.0000 | exact binomial | 0.29 |   
| A | CNN vs ANN | 1.0000 | exact binomial | 0.29 |   
| A | LSTM vs ANN | 1.0000 | exact binomial | -0.29 |   
   
Statistically significant pairwise differences (p < 0.05) are observed for LSTM vs KNN on target G (p = 0.0196), as well as CNN vs RNN and CNN vs ANN on target C (p = 0.0156).  
**H3: Fault Reporting Latency**  
Wilcoxon signed-rank test on paired per-scenario latency across the common 339 rows (p < 0.000001 across all pairs):  
| | | |  
|-|-|-|  
| **Comparison** | **Hodges-Lehmann Estimate (ms)** | **95% Confidence Interval** |   
| CNN vs DTR | 0.0849 | [0.0826, 0.0872] |   
| LSTM vs DTR | 361.2629 | [344.9700, 381.9934] |   
| CNN vs KNN | 0.0653 | [0.0629, 0.0677] |   
| LSTM vs KNN | 361.2436 | [344.3834, 381.2091] |   
| CNN vs RNN | 0.0849 | [0.0826, 0.0873] |   
| LSTM vs RNN | 361.2630 | [344.5830, 382.1831] |   
| CNN vs ANN | 0.0848 | [0.0823, 0.0871] |   
| LSTM vs ANN | 361.2628 | [344.6989, 380.9305] |   
   
LSTM displays a significantly higher latency footprint compared to all non-sequential and standard recurrent baselines (\approx 361\text{ ms} median difference), whereas CNN's latency overhead remains minimal relative to DTR, KNN, RNN, and ANN.  
**H4: Robustness (TOST Equivalence)**  
Two one-sided tests for equivalence (TOST) with an equivalence margin of 5.0 accuracy points:  
| | | | | |  
|-|-|-|-|-|  
| **Model** | **Target** | **1% vs 5% Noise** | **1% vs 10% Noise** | **1% vs 20% Noise** |   
| ANN | G | 0.0043, equivalent | 0.0101, equivalent | 0.0349, equivalent |   
| ANN | C | 0.0069, equivalent | 0.0326, equivalent | 0.0901, not equivalent |   
| ANN | B | 0.0107, equivalent | 0.0481, not equivalent | 0.1214, not equivalent |   
| ANN | A | 0.0060, equivalent | 0.0200, equivalent | 0.0581, not equivalent |   
| CNN | G | 0.0069, equivalent | 0.0273, equivalent | 0.0784, not equivalent |   
| CNN | C | 0.0242, equivalent | 0.0538, not equivalent | 0.1104, not equivalent |   
| CNN | B | 0.0242, equivalent | 0.0654, not equivalent | 0.1388, not equivalent |   
| CNN | A | 0.0135, equivalent | 0.0474, equivalent | 0.1036, not equivalent |   
| Decision Tree | G | 0.0757, not equivalent | 0.1397, not equivalent | 0.2294, not equivalent |   
| Decision Tree | C | 0.0910, not equivalent | 0.2003, not equivalent | 0.3338, not equivalent |   
| Decision Tree | B | 0.0788, not equivalent | 0.1654, not equivalent | 0.2787, not equivalent |   
| Decision Tree | A | 0.0743, not equivalent | 0.1324, not equivalent | 0.2240, not equivalent |   
| K-NN | G | -0.0304, equivalent | -0.0350, equivalent | -0.0220, equivalent |   
| K-NN | C | 0.0078, equivalent | 0.0214, equivalent | 0.0523, not equivalent |   
| K-NN | B | 0.0038, equivalent | 0.0140, equivalent | 0.0401, equivalent |   
| K-NN | A | 0.0028, equivalent | 0.0105, equivalent | 0.0327, equivalent |   
| LSTM | G | -0.0007, equivalent | -0.0008, equivalent | 0.0005, equivalent |   
| LSTM | C | 0.0016, equivalent | 0.0138, equivalent | 0.0358, equivalent |   
| LSTM | B | -0.0019, equivalent | -0.0005, equivalent | 0.0135, equivalent |   
| LSTM | A | 0.0057, equivalent | 0.0082, equivalent | 0.0150, equivalent |   
| RNN | G | -0.0003, equivalent | 0.0057, equivalent | 0.0339, equivalent |   
| RNN | C | 0.0272, equivalent | 0.0851, not equivalent | 0.1781, not equivalent |   
| RNN | B | 0.0039, equivalent | 0.0292, equivalent | 0.0978, not equivalent |   
| RNN | A | 0.0036, equivalent | 0.0201, equivalent | 0.0764, not equivalent |   
   
   
LSTM maintains strict performance equivalence across every target up to 20% noise (p_{\text{TOST}} < 0.05). K-NN maintains equivalence across most conditions except target C at 20% noise. Decision Tree fails equivalence at every noise level and target. ANN, CNN, and RNN maintain equivalence at lower noise thresholds (1%–5%) but fail equivalence as noise reaches 10%–20% on targets C and B.  
