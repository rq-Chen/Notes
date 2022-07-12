# Statistics for Neural Data

## The Multiple Comparison Problem

There are mainly four rational decision making strategies:

- Neyman-Pearson statistics
- Permutation-based statistics
- False-discovery rate control
- Baysian statistics

And the first two are mostly used.

### The Neyman-Pearson's Approach

#### Controlling the error rate

For normal (single-event) tests:

1. Formulate a *null hypothesis* for the value of the population parameters, usually with some assumption, e.g. normality and homogeneity
2. Take some statistics
3. Calculate a critical value given the false alarm rate and the null distribution

In case of the multiple comparison problem (MCP), the above method will generate a proportion of $\alpha$ false positive results. Therefore, we may want to control the family-wise error-rate (FWER) instead:

- FWER is the probability of rejecting at least one true null hypothesis in all our tests.
- A *weak control* of FWER only controls the error rate when *all* null hypotheses are true
- A *strong control* of FWER controls the error rate with any configuration of true and false null hypotheses

There are several ways to acheive this goal.

#### Using group statistics (e.g. maximum T)

The pipeline is modified as follows:

1. Formulate a null hypothesis with *all* elements (e.g. time * freq * channel) in MCP as parameters, (e.g. all of them are the same for all conditions)
2. Take some statistics which depend on *all* data (e.g. the maximum of T)
3. Calculate the critical value with this statistics, and select the significant comparisons all using this value.

The rationale:

- If for all timepoints the null hypothesis is true (namely the H0 is falsely rejected), then there will be a chance of $\alpha$ that the maximum T over all elements will exceed $T_0$ (the critical value). In other words, there is only a chance of $\alpha$ that at least one test in the family is false positive.
- For any configuration of ture and false null hypotheses $\mathcal{T}$ and $\mathcal{ F}$, $P\{\text{max in }\mathcal{T} > T_0 | \mathcal{T}\text{ are ture}\} \le P\{\text{max in }(\mathcal{T}\cup\mathcal{F}) > T_0 | (\mathcal{T}\cup\mathcal{F}) \text{ are ture}\}$ (because adding whatever to the data will not make the maximum smaller), which is $\alpha$. Therefore, maximum T provides strong control over FWER (and with more power than Bonferrroni correction).

Constraints:

- This null hypothesis relies on some assumptions (e.g. Gaussian Random Field Theory for maximum T statistics).
- Low power sometimes (but better than Bonfferoni correction).

#### Bonfferoni correction:

- Divide $\alpha$ by *n* (the number of comparisons) so that totally there is still a chance of $\alpha$ for a single false positive outcome in all *n* tests.
  - Boole's inequality: $P\{\bigcup_{i = 1}^{n} A_i\} \le \sum_{i = 1}^n P\{A_i\}$
  - The probability of rejecting at least one true null hypothesis is no larger than the $\alpha/n * n = \alpha$
- Bonferroni correction is the standard multiple test correction method, which provide strong control of FWER. However, in many cases it is too strict. Besides, it ignores the correlation in data (e.g. between consecutive timepoints), thus having very low statistical power.
- How to report:
  - For simplicity, the way to report Bonferroni-corrected results is usually to report a "corrected" p-value (instead of alpha), which is simply the original p-value multipled by the number of tests
  - Note: if the "corrected" p-value is larger than 1 it will be set as 1


### False Discovery Rate (FDR):

Developed by Benjamini and Hochberg in 1995.

Concept:

- Uncorrected: $\alpha = \langle\frac{\text{number of false postive classifications (FP)}}{\text{All classifications}}\rangle$
- FWER correction: $\alpha = \text{Pr}\{\text{FP} > 0\}$
- FDR correction: $\alpha = \langle\frac{\text{FP}}{\text{Postive classifications (P)}}\rangle$

Differences with FWER:

- FWER expects a chance of $1-\alpha$ that all discoveries are ture
- FDR expects only a proportion $1-\alpha$ of all discoveries are true

Properties:

- Weak control of FWER:
  - if all null hypotheses are true, then any discoveries will all be false alarms
  - therefore, $\text{FDR} = \text{Pr}\{V = 0\} * 0 + \text{Pr}\{V > 0\} * 1 = \text{FWER}$
  - however, if any null hypothesis is true, FDR will be no larger than FWER
- Higer power compared to strong control of FWER
- Limitation:
  - standard FDR (Benjamini–Hochberg procedure) assumes that the data are either uncorrelated or positively correlated
  - not true if the data contain both positive and negative wave, or channel dipoles, etc.
  - a version that is compatible with *negatively correlated data* is **Benjamini and Yekutieli (2001)** (as implemented in fieldtrip).

## Permutation-based Approach (non-MCP)

Actually the procedure is similar to the previous approach, except that it uses a model-free way to determine the threshold (critical value).

1. Formulate a null hypothesis for the probability distributions of the observations (e.g. distributions under all conditions are the same) without really specifing the form of it.
2. Take some statistics.
3. Calculate the critical value.

Example of a within subject design:

<img src = "Permutation test for within subject design.png" style = "zoom:30%" />

The null hypothesis is that the distribution of the random variable D under condition A is the same as the one under condition B. Under this assumption, the redrawing samples will have exactly the same distribution as the unknown underlying distribution of *D*. Therefore, you can sample from it (practically, simply calculating the statistics for each of the redrawed samples) to get the distribution of your statistics.

Therefore, you can calculate some statistics *S* for each permuted observation just like sampling from a uniform distribution. You can plot the histogram of *S* and see what percentile the real data is at, say the top 5%, then you can safely reject the null hypothesis without any extra assumptions.

One more hints: $$D_{Ai}$$ or $$D_{Bi}$$ here can be multidimensional, e.g. a time * freq * channel matrix resulted in averaging over trials for subject *i* under condition *A* or *B*. You should only reassign (relabel) the condition (for multiple subjects) or trials (for single subject) for permutation, rather than shuffling the time course within each trial.

Detailed pipeline:

1. Collect the trials of the two experimental conditions in a single set.

2. Randomly draw as many trials from this combined data set as there were trials in condition 1 and place those trials into subset 1. Place the remaining trials in subset 2. The result of this procedure is called a random partition.

3. Calculate the test statistic on this random partition.

4. Repeat steps 2 and 3 a large number of times and construct a histogram of the test statistics.

5. From the test statistic that was actually observed and the histogram in step 4, calculate the proportion of random partitions that resulted in a larger test statistic than the observed one. This proportion is called the p-value.

6. If the p-value is smaller than the critical alpha-level (typically, 0.05), then conclude that the data in the two experimental conditions are significantly different.

## Cluster-level Permutation

- The aforementioned method is only at the single point level and you can also perform the MCP correction by maximum T
  - strong control as Bonferroni
  - moderate power between Bonfferoni and FDR
- However, in order to make better use of the structure of data, it's better to focus on *clusters* rather than individual tests:
  - sample-level: $P(\text{at least one }t > t_0 | H_0) = \alpha$, where $t$ is a statistics calculated from each test
  - cluster-level: $P(\text{at least one ClusterStat} > \text{ClusterStat}_0 | H_0) = \alpha$, where $\text{ClusterStat}$ is a statistics calculated from each *cluster*
  - clusters can be defined by thresholding
  - $\text{ClusterStat}_0$ can be found by the distribution of  maximum ClusterStat (similar to the idea of maximum T)

Here is an example, where the threshold is defined by an uncorrect *t* statistics:

<img src = "Permutation test for ERF.png" style = "zoom:30%" />

A detailed pipeline:

<img src = "Cluster level permutation test.png" style = "zoom:30%" />

And we should add a final step: find the $(1 - \alpha) \times 100 \%$ percentile of the maximum cluster-level statistics as the threshold for "significant" clusters, and report the clusters in the observed data (not just the largest one!) with a cluster-level statistics larger than it.

Note that if you have multiple conditions instead of one, you may need to use *F* statistics instead of *t*. In other words, every time you draw a set of sample, you do an ANOVA at eatch datapoint (time or time * freq * position, etc.) to get a curve of F statistics over time (or time * freq * location, etc.), then you threshold it by the 95 percentile of *F* (like an ANOVA for a single timepoint) and calculate the suprathreshold area. While in the two-condition case, you are essentially calculating the curve of *t* statistics over time (or time * freq * location, etc.).

**(IMPORTANT)** how to interpret clusters:

- For the whole dataset:

  - The presence of any clusters that survive the test tells you that your data is different among conditions somewhere at some time in some frequency band
  - Likewise, if you only do the cluster-based permutation tests in a proportion of your data, you can safely said the the data is different among conditions within this proportion (however, exactly when and where within this proportion is still unknown)

- For clusters:

  - A cluster that survives the test is NOT necessarily different among conditions.

  - That's because the $H_0$ for permutation test is just that the data comes from a same distribution (under which assumption you can draw from resampling distribution just like uniform distribution, and get the distribution of whatever statistic you like - sample level or cluster level), it does NOT specify any time or space

  - If we reject $H_0$, the only thing we can safely say is that the data is different among the conditions in somewhere at sometime for some frequency.

  - One confounding factor is that the cluster-level statistics may be more sensitive to some particular aspect or porportion of data, so that the "significant" cluster will be influenced.

    - If you test that the physical dimensions of male bodies are different from those of female bodies, you will likely find that the H0 (the physical dimensions of male and female bodies come from the same probability distribution) will be rejected in favour of the alternative (they come from different distributions).

      However, from this result, one cannot conclude that men and women have different foot sizes. In fact, it may be that the test statistic that was used to compare male and female bodies was sensitive to other aspects than foot size.

    - This also means that you can define better statistics to test for specific aspect in the data.

- Besides, if a cluster is "significant", it does NOT mean that the data are significantly different at a certain timepoint within that cluster

  - because the threshold is arbitary, and one timepoint may just be lucky that it lies in a cluster that contains a lot of "really" significant timepoints

- See the [original paper](https://www.sciencedirect.com/science/article/pii/S0165027007001707?via%3Dihub) on this test, the [Fieldtrip suggestion](http://www.fieldtriptoolbox.org/faq/how_not_to_interpret_results_from_a_cluster-based_permutation_test/), and Steven Luck's suggestion (ERP book, [online chapter 13](https://erpinfo.org/s/Ch_13_Mass_Univariate_and_Permutations.pdf), the last section *Practical Considerations*)

### Threshold-Free Cluster Enhancement (TFCE)

- Concept:

  - In cluster-mass based permutation tests, the choice of cluster-forming threshold is quite arbitary. However, is is important andt it will influence the extent of the cluster.
  - Can we sum up the "cluster mass" with all possible thresholds to characterize how wide and peaky a cluster is?

- Cluster enhancement:
  $$
  \text{TFCE}(p) = \int_0^{h_p} e(h)^Eh^Hdh
  $$
  is the TFCE value for a specific datapoint *p*, where $h_p$ is its statistic (e.g. t-value or F-value), $e(h)$ is the extent of the cluster, with a specific threshold $h$, and parameters $E$ and $H$ control the relative importance of cluster extent and statistical value.

  <img src="https://benediktehinger.de/upload/TFCEcalculation.gif" alt="img" style="zoom:33%;" />

  The figure above illustrates the situation where parameters $E$ and $H$ are both set to 1. However, in practice they are often set as $E = 0.5, H = 2$ due to certain reasons.

- Thresholding:

  - After calculating the TFCE value for each datapoint, we can use permutation to get the null distribution of the maximum of TFCE over all datapoints
  - Then we threshold the empirical TFCE by the $(100 - \alpha)$ percentile of the distribution and get the suprathreshold clusters

- Interpretation:

  - Note that just like regular cluster-based permutation tests, there is no guarantee that any point within the suprathreshold clusters is significant

## Unordered Notes

- Interquartile range (IQR): range of the middle 50% (25%-75% percentile) of the data
- MANOVA is **NOT** ANOVAN! The former analyzes multiple *dependent* variables and the latter analyzes multiple *independent* variables.
- How to characterize the extent to which the data is spatially clustered: **Moran's I** statistics.
- The inverse of *covariance matrix* is called *precision matrix* and it is widely use in Bayesian statistics since it has an analytic conjugate distribution (Gamma distribution).
- How to sample from any distribution: calculate the cumulative distribution $$F(x)$$, generate a uniform random number $$x_0$$ between 0-1, and return $$y$$ where $$F(y) = x_0$$.
  - In this way, $$P\{y_0 < y \le y_1\} = F(y_1) - F(y_0) = \int_{y_0}^{y_1} f(x) dx$$
  - Practically, you may need to sample $$F(x)$$ and return an approximated solution by `find(x0 > samp, 1, 'first')`
- Bootstrapping: using the winthin-sample distribution as the estimation of the population
  - Usage: usually used when you only have one sample, or when the underlying distribution for your statistics is unknown, so you use bootstrapping to calculate the standard error of mean for your statitics
  - resample the data by random sampling with replacement, and calculate the MSE

## Discriminant Analysis

- When the data is labeled, we should use discriminant analysis (e.g. LDA, canonical discriminant analysis) rather than PCA for dimensionality reduction