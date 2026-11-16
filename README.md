Domain: E-commerce

## How to run

This project answers the order-value, churn-retention and customer-segmentation questions in the Flipkart capstone scenario. All records are synthetic; no real customer or company data is used.

Tested with Python 3.12.15 and NumPy 2.2.6. From a terminal with Python 3.10 or newer, run these commands in order; the activation command below is for Linux/macOS.

```bash
git clone https://github.com/JAIkaranBHT/flipkart-ecommerce-decision-intelligence.git
cd flipkart-ecommerce-decision-intelligence
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
python -m unittest -v
```

On Windows, replace the activation command with `.venv\Scripts\activate`. The generator must print `800 10 170 c6f249ba693f1ebeed318d6d5e18de23`; it is committed without changes from the supplied brief.

The repository contains the four entry points, the generated `data/records.csv`, and all twelve required text/CSV files in `outputs/`. `common.py` handles file loading and output formatting; `models.py` contains the classification routines shared by Parts 2 and 4, without running either Part on import. `test_project.py` checks the calculations, pinned conventions and committed deliverables.

All graded algorithms are implemented with Python and NumPy arithmetic. The multiple-regression normal equation uses `numpy.linalg.solve`, and PCA uses `numpy.linalg.eigh`; no ready-made estimator or metric function is used. Calculations keep full precision until values are written, with sample SDs throughout and four decimal places for floating-point outputs.

## Part 1 findings

The value column is measured in hundreds of INR, not individual Rupees. The following results come from `outputs/part1_results.txt` and use all 800 records.

- **Planning claim (R1.1):** The estimated mean is 8.9382, with sample SD 9.1723, standard error 0.3243 and a 95% interval of [8.3026, 9.5738]. Reject the claim that the population mean is 8.0: `z_stat = 2.8930`, whose magnitude exceeds 1.96. A Type I error here would mean concluding that the population average order value differs from INR 800 when it actually equals INR 800. The observed gap is 0.9382 column units, about INR 93.82 or 11.73% of the planning assumption; that is potentially meaningful for revenue planning, although its business importance depends on margins and volume.
- **Future sample (R1.2):** `n_required = 1293`, compared with 800 available records. The team would need 493 additional observations to target a 95% margin of INR 50, assuming the present sample SD remains suitable.
- **Regression and optimisation (R1.3):** The closed-form line is approximately `value = -3.6598 + 0.6701 × X4`, with R² 0.5196, RMSE 6.3537 and MAE 3.7636 on the original scale. TSS 67221.4285 equals ESS 34925.6088 plus RSS 32295.8197 up to rounding, and `gd_slope_gap = 0.0000`, below the required 0.001. Standardising X4 and taking 500 full-batch updates at learning rate 0.1 brings gradient descent to the same convex squared-error minimum as the closed-form fit.
- **Residual funnel (R1.4):** `resid_sd_ratio_raw = 3.7525`: the upper fitted-value half has much greater residual spread than the lower half. Regressing ln(value) on ln(X4) reduces that ratio to `resid_sd_ratio_log = 1.0396`, largely removing the funnel by this diagnostic, though not proving constant variance everywhere. Raw and logged R² values describe different response scales and are not compared as a model-selection score.
- **Extra predictors (R1.5):** Applying the requested absolute-coefficient rule above 0.05, excluding the constant, selects ln(X4), past average order value (1.1576); X2, complaint count (0.0940); X3, discount-order share (0.1067); and X6, cash-on-delivery flag (0.1518). Multiple-regression adjusted R² is 0.7403 versus 0.6561 for the simple log-log model, so the extra predictors earn their place by this adjusted in-sample criterion. These coefficients use different units; the mechanical list is neither an importance ranking nor a test of statistical significance.

## Part 2 findings

The pinned stratified split gives 560 training records with 119 churners and 240 test records with 51 churners. All seven features, including the flags, use training means and sample SDs; neither the test records nor their labels influence fitting, scaling, K selection or threshold selection.

- **Learning rate (R2.1):** At 200 updates, losses from lowest to highest are η = 1.0: 0.3703, η = 0.1: 0.3776, and η = 0.01: 0.5091. The smallest rate progresses slowly, while the largest tested rate converges faster here; this result does not establish that an arbitrarily larger rate would remain stable.
- **Predictors and odds (R2.2):** The three largest absolute weights are X3, discount-order share (0.9013); X1, tenure (-0.5554); and X2, complaint count (0.5505). Holding the other predictors fixed, a one-training-SD increase in discount-order share multiplies the modelled odds of churn by 2.4629, the largest odds ratio. This is a model association, not evidence that discounts cause churn.
- **Accuracy and threshold (R2.3):** Predicting no churn for everyone gives `blind_accuracy = 0.7875`, close to logistic accuracy 0.8042 at τ = 0.5, so accuracy hides the cost of missed churners. Lowering τ from 0.5 to 0.3 raises recall from 0.4706 to 0.7451 but lowers precision from 0.5455 to 0.4634. The confusion cells change from TP/FP/TN/FN = 24/20/169/27 to 38/44/145/13.
- **Logistic versus KNN (R2.4):** Five-fold mean F1 selects K = 5, with mean CV F1 0.4541. On the test records, logistic at τ = 0.3 has F1 0.5714 and recall 0.7451, compared with KNN F1 0.3696 and recall 0.3333. Each fold uses the specified training-position-modulo-five rule and the already fitted training standardisation; equal neighbour distances retain reference file order.
- **Retention decision (R2.5):** Choose logistic regression with τ = 0.15 as the production candidate under the supplied cost rule, INR 50 per false alarm and INR 400 per missed churner. It minimises the training-grid cost at INR 12150; its held-out cost is INR 6050, versus INR 11800 at τ = 0.5 and INR 20400 when nobody is flagged. Selecting τ on training records preserves the test records as an evaluation set rather than tuning the decision to their outcomes.

`outputs/part2_results.txt` contains the losses, coefficients, odds ratios, metrics and costs. `outputs/part2_thresholds.csv` contains the nine test ROC points, while `outputs/part2_knn_cv.csv` contains all eleven candidate K scores.

## Part 3 findings

Segmentation uses only the five continuous predictors, standardised across all 800 records. Churn and value are added only after clustering to describe the groups; neither is used in the clustering distances.

### Segment count and profiles

The successive WCSS drops for K = 2→3, 3→4, 4→5 and 5→6 are 659.8282, 108.0739, 171.0575 and 45.5466 respectively, calculated before rounding. The prescribed elbow rule selects K = 3 because 108.0739 is less than half of 659.8282. The later drops do not shrink steadily: the pinned starting rows produce reproducible local solutions, not necessarily global optima.

The K = 3 profiles below are ordered by increasing discount-order share, as required. All figures are from `outputs/part3_segments.csv`; past and current order values remain in hundreds of INR.

| Segment and business name | Size | Tenure | Complaints | Discount share | Past order value | Loyalty | Churn rate | Current value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: Established low-friction shoppers | 322 | 47.5028 | 1.9503 | 0.2250 | 12.5301 | 62.1584 | 0.0186 | 3.5976 |
| B: Higher-spend shoppers with friction | 255 | 28.6071 | 6.0706 | 0.4534 | 28.7192 | 55.0082 | 0.2353 | 17.1248 |
| C: Newer discount-heavy shoppers | 223 | 14.3892 | 4.2287 | 0.7118 | 16.5099 | 38.5215 | 0.4664 | 7.2883 |

- **Segment A action (R3.2):** Offer a lightweight loyalty check-in instead of a blanket retention coupon. This group has the longest tenure, highest loyalty score and lowest observed churn rate.
- **Segment B action (R3.2):** Prioritise complaint resolution before another promotion. Its higher past order value and complaint count make service recovery a more grounded first action than assuming price is the main issue.
- **Segment C action (R3.2):** Test an onboarding and targeted-retention offer with a control group. C has the highest churn rate, 0.4664, alongside shorter tenure, lower loyalty and the highest discount share; the profiles do not establish seasonality or coupon effectiveness.

### Hierarchical cuts

Single linkage cuts into four clusters: {E1}, {E2, E3, E6}, {E4}, {E5}; its largest height gap is between 1.4000 and 2.8000. Complete linkage cuts into two clusters: {E1, E4, E5} and {E2, E3, E6}; its largest gap is between 2.9000 and 7.9000.

Single linkage uses the closest member pair, so a short bridge can make the final merge relatively cheap and put the largest gap earlier. Complete linkage uses the farthest pair, so the two final groups remain separated until height 7.9000 and the largest-gap cut comes later. Average linkage also gives two clusters at its largest gap; all fifteen merges are recorded in `outputs/part3_hierarchical.csv`.

### Principal components

The mean vector after standardisation is (0, 0, 0, 0, 0), up to floating-point precision. The code explicitly centres it again, forms the sample covariance, decomposes it, sorts the eigenpairs, fixes their signs, retains the top two vectors and projects the centred rows.

PC1 loadings on X1 through X5 are 0.5317, -0.4100, -0.5187, -0.2916 and 0.4418. Longer tenure and higher loyalty drive positive scores, while higher discount share, complaints and past order value drive negative scores; tenure and discount share have the largest absolute loadings. The eigenvalues sum to 5.0000; two components retain 0.7593 of variance, so three are required to reach at least 80%, retaining 0.8616.

Mean PC1 scores are A: 1.7773, B: -0.7831 and C: -1.6709. This orders the established, low-friction group above the more discount-heavy, shorter-tenure group, while B lies between them; a one-dimensional summary does not capture every difference between the profiles. Loadings, variance ratios and the first three row projections are in `outputs/part3_results.txt`.

## Part 4 findings

### Splits and model combinations

S4, tenure ≤ 24 months, is best by both information gain (0.1252) and Gini gain (0.0606), separating training churn rates of 0.4574 and 0.0887. It reduces impurity more than the other three candidates, but growing a tree until every leaf is pure could memorise incidental training patterns. A maximum-depth limit is one way to stop that growth.

The five prescribed comparisons from `outputs/part4_voting.csv` are:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Logistic, τ = 0.3 | 0.4634 | 0.7451 | 0.5714 |
| KNN, K = 5 | 0.4146 | 0.3333 | 0.3696 |
| Stump, τ = 0.3 | 0.4333 | 0.7647 | 0.5532 |
| Hard vote, at least two positive votes | 0.4375 | 0.6863 | 0.5344 |
| Soft vote, mean probability ≥ 0.3 | 0.4368 | 0.7451 | 0.5507 |

Of these model families, choose logistic regression, then use the training-cost-selected τ = 0.15 for the proposed retention policy rather than treating the comparison threshold 0.3 as optimal. Combining models did not improve F1 over logistic at 0.3: hard voting reduced both F1 and recall, while soft voting kept recall at 0.7451 but reduced F1 to 0.5507. The stump catches slightly more churners at 0.7647 recall, but its F1 is lower at 0.5532.

### Boosting and out-of-bag share

After round 1 normalisation, the two misclassified points, 3 and 7, each carry weight 0.2500, for a combined 0.5000. Round 2 trusts h2 more because its weighted error is 0.1667 versus h1's 0.2500, producing α2 = 0.8047 rather than α1 = 0.5493.

The final eight-point ensemble accuracy is 0.7500, so reweighting is not a guarantee of improvement on every example. The out-of-bag probability for one record in a 560-draw bootstrap is 0.3676; these results and the complete weight trail are in the Part 4 result and AdaBoost files.

### Architecture sizing

Model A option 1 has 13002 parameters. For option 2, the output shapes and parameters are below; dense layers are shown as 1 × 1 × width to keep one shape convention, and the input row is included for context.

| Layer | Output shape, height × width × depth | Parameters |
|---|---|---:|
| Input | 28 × 28 × 1 | 0 |
| Conv1, 5 × 5, stride 1, padding 2 | 28 × 28 × 6 | 156 |
| ReLU | 28 × 28 × 6 | 0 |
| Max-pool, 2 × 2, stride 2 | 14 × 14 × 6 | 0 |
| Conv2, 5 × 5, stride 1, padding 0 | 10 × 10 × 16 | 2416 |
| ReLU | 10 × 10 × 16 | 0 |
| Max-pool, 2 × 2, stride 2 | 5 × 5 × 16 | 0 |
| Flatten | 1 × 1 × 400 | 0 |
| Dense, 32 neurons | 1 × 1 × 32 | 12832 |
| ReLU | 1 × 1 × 32 | 0 |
| Dense, 10 class scores | 1 × 1 × 10 | 330 |
| Softmax, proposed output activation | 1 × 1 × 10 | 0 |

The CNN total is 15734, including every bias. For the larger 128 × 128 × 3 input, the specified fully connected layer has 50332672 parameters, compared with 1792 for the convolutional layer; convolution reuses each filter's weights across spatial locations rather than assigning a new weight to every input-output pair.

For Model B, using column vectors, W_xh is 128 × 64, W_hh is 128 × 128 and W_hy is 16 × 128; the hidden and output biases have lengths 128 and 16. Its total is 26768 parameters and does not grow with sequence length because the same recurrent weights are reused at each step, although computation grows.

For the 10-category image classifier, use ReLU hidden activations and softmax outputs with multiclass cross-entropy loss; no neural network is trained in this project. In a very deep plain network, repeated multiplication through small derivatives can shrink gradients until early layers barely update. A skip connection adds a direct path for information and gradients, reducing the need for every update signal to traverse the full stack.

## Model-choice memo

To: Head of the retention team

Choose logistic regression as the structured-data production candidate, using the training-selected alert threshold of 0.15. In `outputs/part2_results.txt`, this policy costs INR 6050 on the held-out records, compared with INR 11800 for logistic at 0.5 and INR 20400 for flagging nobody. These costs apply the brief's INR 50 false-alarm and INR 400 missed-churn rule; they are not measured campaign returns.

The model also offers a simpler explanation and deployment path than maintaining three learners. In `outputs/part4_voting.csv`, logistic at the comparison threshold 0.3 has F1 0.5714 and recall 0.7451; hard voting gives 0.5344 and 0.6863, while soft voting gives 0.5507 and 0.7451. Those comparison scores are not scores at the recommended 0.15 threshold.

The threshold was selected from training costs, not from test performance. The evidence is nevertheless synthetic, based on one fixed split with only 51 positive test records, and does not demonstrate that a retention coupon changes behaviour. Before a real rollout, validate on recent authorised customer records, check capacity and calibration, and run a controlled pilot to measure incremental retained margin.

Neither Model A for product images nor Model B for review topics is worth building before the structured retention model is in production: neither has labelled task data or a demonstrated benefit to this decision yet.

## Limitations

- **Synthetic setting:** The generator invents every record. Results describe this dataset, not Flipkart's actual customers, economics or performance.
- **Single evaluation split:** The fixed file-order split is required for reproducibility, not a substitute for repeated or time-based validation. The 240-row test set contains only 51 churners, making recall and cost sensitive to a few records.
- **Selection optimism:** The cost threshold is optimised on the same training rows used to fit logistic regression. This follows the brief but can be optimistic; a separate validation set would strengthen a real deployment decision.
- **Prescribed CV scaling:** KNN cross-validation uses scaling fitted on all 560 training rows, as required, so each held-out fold contributes to those scaling statistics. A usual leakage-stricter CV pipeline would refit scaling within each fold; neither approach here uses the final test records.
- **Simplified economics:** The cost rule omits coupon costs for true positives, contact limits, customer lifetime-value differences and whether an intervention prevents churn. Lower classification cost is not proof of causal campaign profit.
- **Descriptive models:** Regression fit is in-sample, segments depend on the pinned initial centroids, and PCA maximises variance rather than retention value. None establishes causal effects or guarantees stable groups on new records.
- **Architecture counts only:** Image and text models are sized, not trained or benchmarked. Their parameter totals do not establish accuracy, latency or business value.
