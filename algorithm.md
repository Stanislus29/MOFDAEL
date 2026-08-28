We test, computationally, the following models generated. 

The problem is a classification problem where we are to train a model to classify faults on a three phase electrical lines: G,C,B,A. Where G is ground/earth. We train a number of models using different techniques. ML: svm, decision tree, knn; DL; CNN. 

The approach is to train sub-models under each approach to identify faults along a particular line, such that:

$$
m_\text{G(t)}, m_\text{C(t}, m_\text{B(t}, m_\text{A(t} \elementof M \tag{1.0} \\

: t \elementof {t_\text{(svm)}, t_\text{dtr}, t_\text{knn}, t_text{cnn}
$$

The function under 1.0 is performed recursively for all $t$ 

Each model produces its accuracy, precision, robustness, f1 score within its own train script

## Tests

We test the resulting models against the full dataset, and record their predictions against the true value in the dataset. That true value will serve as our relay. For each prediction, we also record the latency for each model's prediction. 

```
data = dataset

svm = svm/models/svm.joblib
dtr = dtr/models/dtr.joblib
knn = knn/models/knn.joblib
cnn = cnn/models/cnn.joblib

result = pd.createdf
models = [svm, dtr, knn, cnn]

for m in models.items(): #for each model
    m = model.initialize(data) 
    time_start = time.time
    
    inference = m.predict(X) #run a single pass prediction on the dataset
    
    latency = time.time.time_start
    
    return inference, latency #return its result
        
    where in inference: G, C, B, A in inference(X) #find where in the inference the line headers are
        items.append(item_m) #append the headers for the lines to be in the form of G_{model_type}, A _{model_type} etc
        return inference 

result = df.append_right(inference, latency) #append the result for the latest model to the right of the dataframe
```

Follwing, we then perform these tests:

## H₁ Fault detection accuracy. 
McNemar’s test is applied to compare each machine learning and deep learning model against the conventional relay baseline. For two classifiers evaluated on a common test set, McNemar’s test examines the null hypothesis of marginal homogeneity, conditioning on the discordant scenarios those on which exactly one method is correct. Scenarios classified identically by both methods are uninformative regarding their relative performance and are appropriately excluded from the test statistic (Dietterich, 1998). The exact binomial form is used where the number of discordant pairs is below 25, and the continuity-corrected chi-square form otherwise. Cochran’s Q is reported as an omnibus test across all eight methods (Cochran, 1950). Effect size is reported as the odds ratio derived from the discordant cells, together with the paired difference in accuracy in percentage points and its confidence interval.
## H₂ Fault classification performance. 
Cochran’s Q is applied as an omnibus test across the seven learned models, examining whether classification accuracy differs across methods for correlated binary outcomes. Where the omnibus test is significant, pairwise McNemar tests are conducted for each deep learning model against each classical machine learning model, yielding twelve cross-family comparisons. Aggregation of models into family-level means is not performed, as averaging across a small number of heterogeneous models suppresses between-model variance and yields an effective sample size too small for meaningful inference. As a supplementary analysis, the Friedman test with post-hoc Wilcoxon signed-rank comparisons is applied to fold-level macro-averaged F1 scores, following the protocol established for classifier comparison by Demšar (2006).
## H₃ Fault reporting latency. 
Processing latency is measured per scenario and compared using the Wilcoxon signed-rank test. The non-parametric procedure is adopted because timing distributions are characteristically right-skewed, violating the normality assumption of the paired t-test. Effect size is reported as the matched-pairs rank-biserial correlation, accompanied by the Hodges–Lehmann estimate of the median difference and its distribution-free confidence interval. The test is conducted two-sided, and the direction of the observed difference is reported without adjustment to the hypothesised direction.
H₄ Robustness under varying operating conditions. 
Because H₄ asserts stability rather than difference, conventional null-hypothesis significance testing cannot provide supporting evidence: failure to reject the null indicates only that a difference was not detected, a result equally consistent with insufficient statistical power (Lakens, 2017). Two one-sided tests (TOST) are therefore applied to establish statistical equivalence.


The pseudocode is just a baseline sketch to show the preferred architecture. All tests are performed by model, for each line. Always load the latest model from the folder. 
