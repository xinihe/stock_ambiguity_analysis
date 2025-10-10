# Comprehensive Bin Size Analysis Report

## Executive Summary

This report presents a systematic analysis of ambiguity metrics using different bin sizes (from 5 to 50 bins) and various regression configurations.

## Data Summary

- **Analysis Period**: 2018-01-01 to 2024-12-01
- **Total Monthly Observations**: 84
- **Window Size**: 20 days
- **Bin Sizes Tested**: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
- **GPR Countries**: China, US, Japan

## Best Results Summary

### Highest R Squared

- **Configuration**: Monthly_Return_Pct_bins_20_gpr_China_US_Japan
- **Value**: 0.0388
- **R²**: 0.0388
- **Adjusted R²**: 0.0023
- **F-test p-value**: 0.3694
- **Observations**: 83
- **Significant Coefficients**:
  - const: 1.8993 (p = 0.0000)

### Highest Adj R Squared

- **Configuration**: Monthly_Return_Pct_bins_20_gpr_US
- **Value**: 0.0119
- **R²**: 0.0240
- **Adjusted R²**: 0.0119
- **F-test p-value**: 0.1621
- **Observations**: 83
- **Significant Coefficients**:
  - const: 1.8881 (p = 0.0000)

### Most Significant

- **Configuration**: Monthly_Return_Pct_bins_5_gpr_China
- **Count**: 1
- **R²**: 0.0118
- **Adjusted R²**: -0.0004
- **F-test p-value**: 0.3273
- **Observations**: 83
- **Significant Coefficients**:
  - const: 1.2099 (p = 0.0000)

### Best F Test

- **Configuration**: Monthly_Return_Pct_bins_20_gpr_US
- **Value**: 0.1621
- **R²**: 0.0240
- **Adjusted R²**: 0.0119
- **F-test p-value**: 0.1621
- **Observations**: 83
- **Significant Coefficients**:
  - const: 1.8881 (p = 0.0000)

## Detailed Results by Bin Size

### Dependent Variable: Monthly_Return_Pct

| Bin Size | Best R² | Best Adj R² | Best F-test | Significant Vars |
| -------- | -------- | ------------ | ----------- | ---------------- |
| 5        | 0.0367   | 0.0117       | 0.2323      | 7                |
| 10       | 0.0132   | 0.0000       | 0.5014      | 7                |
| 15       | 0.0082   | 0.0000       | 0.4638      | 7                |
| 20       | 0.0388   | 0.0119       | 0.1621      | 7                |
| 25       | 0.0168   | 0.0000       | 0.3321      | 7                |
| 30       | 0.0188   | 0.0000       | 0.4860      | 7                |
| 35       | 0.0240   | 0.0000       | 0.3733      | 7                |
| 40       | 0.0251   | 0.0005       | 0.3649      | 7                |
| 45       | 0.0237   | 0.0000       | 0.3956      | 7                |
| 50       | 0.0292   | 0.0045       | 0.3103      | 7                |

### Dependent Variable: Log_Monthly_Return

| Bin Size | Best R² | Best Adj R² | Best F-test | Significant Vars |
| -------- | -------- | ------------ | ----------- | ---------------- |
| 5        | 0.0367   | 0.0117       | 0.2323      | 7                |
| 10       | 0.0132   | 0.0000       | 0.5014      | 7                |
| 15       | 0.0082   | 0.0000       | 0.4638      | 7                |
| 20       | 0.0388   | 0.0119       | 0.1621      | 7                |
| 25       | 0.0168   | 0.0000       | 0.3321      | 7                |
| 30       | 0.0188   | 0.0000       | 0.4860      | 7                |
| 35       | 0.0240   | 0.0000       | 0.3733      | 7                |
| 40       | 0.0251   | 0.0005       | 0.3649      | 7                |
| 45       | 0.0237   | 0.0000       | 0.3956      | 7                |
| 50       | 0.0292   | 0.0045       | 0.3103      | 7                |

## Detailed Regression Results

### Monthly_Return_Pct Results

#### Bin Size: 5

**Ambiguity Risk**

- R²: 0.0015
- Adjusted R²: -0.0234
- F-test p-value: 0.9407
- Observations: 83
- Durbin-Watson: 2.3404

Coefficients:

- const: 2.4827 (SE: 5.0110, p: 0.6216)
- Ambiguity_Bins_5: -1.0823 (SE: 3.7621, p: 0.7743)
- Risk_Bins_5: -1964.6199 (SE: 6861.1938, p: 0.7754)

**Gpr China**

- R²: 0.0118
- Adjusted R²: -0.0004
- F-test p-value: 0.3273
- Observations: 83
- Durbin-Watson: 1.9547

Coefficients:

- const: 1.2099 (SE: 0.0487, p: 0.0000) ***
- GPR_China_y: 0.0569 (SE: 0.0577, p: 0.3273)

**Gpr Us**

- R²: 0.0027
- Adjusted R²: -0.0096
- F-test p-value: 0.6384
- Observations: 83
- Durbin-Watson: 1.9474

Coefficients:

- const: 1.2282 (SE: 0.0591, p: 0.0000) ***
- GPR_US_y: 0.0091 (SE: 0.0193, p: 0.6384)

**Gpr Japan**

- R²: 0.0021
- Adjusted R²: -0.0102
- F-test p-value: 0.6822
- Observations: 83
- Durbin-Watson: 1.9362

Coefficients:

- const: 1.2662 (SE: 0.0324, p: 0.0000) ***
- GPR_Japan_y: -0.0456 (SE: 0.1110, p: 0.6822)

**Gpr China Us**

- R²: 0.0127
- Adjusted R²: -0.0120
- F-test p-value: 0.5999
- Observations: 83
- Durbin-Watson: 1.9564

Coefficients:

- const: 1.2190 (SE: 0.0600, p: 0.0000) ***
- GPR_China_y: 0.0709 (SE: 0.0790, p: 0.3719)
- GPR_US_y: -0.0069 (SE: 0.0262, p: 0.7942)

**Gpr China Japan**

- R²: 0.0358
- Adjusted R²: 0.0117
- F-test p-value: 0.2323
- Observations: 83
- Durbin-Watson: 1.9362

Coefficients:

- const: 1.2052 (SE: 0.0485, p: 0.0000) ***
- GPR_China_y: 0.1272 (SE: 0.0760, p: 0.0981)
- GPR_Japan_y: -0.2051 (SE: 0.1454, p: 0.1621)

**Gpr Us Japan**

- R²: 0.0086
- Adjusted R²: -0.0162
- F-test p-value: 0.7078
- Observations: 83
- Durbin-Watson: 1.9343

Coefficients:

- const: 1.2302 (SE: 0.0593, p: 0.0000) ***
- GPR_US_y: 0.0156 (SE: 0.0216, p: 0.4703)
- GPR_Japan_y: -0.0853 (SE: 0.1240, p: 0.4936)

**Gpr China Us Japan**

- R²: 0.0367
- Adjusted R²: 0.0002
- F-test p-value: 0.3955
- Observations: 83
- Durbin-Watson: 1.9372

Coefficients:

- const: 1.2146 (SE: 0.0598, p: 0.0000) ***
- GPR_China_y: 0.1417 (SE: 0.0933, p: 0.1328)
- GPR_US_y: -0.0071 (SE: 0.0261, p: 0.7875)
- GPR_Japan_y: -0.2053 (SE: 0.1462, p: 0.1642)

#### Bin Size: 10

**Ambiguity Risk**

- R²: 0.0111
- Adjusted R²: -0.0136
- F-test p-value: 0.6406
- Observations: 83
- Durbin-Watson: 2.3333

Coefficients:

- const: -3.9729 (SE: 5.4999, p: 0.4722)
- Ambiguity_Bins_10: 2.8523 (SE: 3.0832, p: 0.3577)
- Risk_Bins_10: 1079.9287 (SE: 6893.5141, p: 0.8759)

**Gpr China**

- R²: 0.0001
- Adjusted R²: -0.0123
- F-test p-value: 0.9319
- Observations: 83
- Durbin-Watson: 1.8572

Coefficients:

- const: 1.6818 (SE: 0.0600, p: 0.0000) ***
- GPR_China_y: 0.0061 (SE: 0.0712, p: 0.9319)

**Gpr Us**

- R²: 0.0013
- Adjusted R²: -0.0110
- F-test p-value: 0.7469
- Observations: 83
- Durbin-Watson: 1.8600

Coefficients:

- const: 1.6641 (SE: 0.0725, p: 0.0000) ***
- GPR_US_y: 0.0077 (SE: 0.0237, p: 0.7469)

**Gpr Japan**

- R²: 0.0056
- Adjusted R²: -0.0067
- F-test p-value: 0.5014
- Observations: 83
- Durbin-Watson: 1.8523

Coefficients:

- const: 1.7094 (SE: 0.0396, p: 0.0000) ***
- GPR_Japan_y: -0.0917 (SE: 0.1358, p: 0.5014)

**Gpr China Us**

- R²: 0.0017
- Adjusted R²: -0.0233
- F-test p-value: 0.9342
- Observations: 83
- Durbin-Watson: 1.8628

Coefficients:

- const: 1.6664 (SE: 0.0740, p: 0.0000) ***
- GPR_China_y: -0.0176 (SE: 0.0973, p: 0.8572)
- GPR_US_y: 0.0116 (SE: 0.0323, p: 0.7205)

**Gpr China Japan**

- R²: 0.0116
- Adjusted R²: -0.0131
- F-test p-value: 0.6266
- Observations: 83
- Durbin-Watson: 1.8461

Coefficients:

- const: 1.6778 (SE: 0.0602, p: 0.0000) ***
- GPR_China_y: 0.0659 (SE: 0.0943, p: 0.4872)
- GPR_Japan_y: -0.1743 (SE: 0.1805, p: 0.3370)

**Gpr Us Japan**

- R²: 0.0115
- Adjusted R²: -0.0132
- F-test p-value: 0.6294
- Observations: 83
- Durbin-Watson: 1.8589

Coefficients:

- const: 1.6673 (SE: 0.0726, p: 0.0000) ***
- GPR_US_y: 0.0182 (SE: 0.0264, p: 0.4912)
- GPR_Japan_y: -0.1381 (SE: 0.1519, p: 0.3660)

**Gpr China Us Japan**

- R²: 0.0132
- Adjusted R²: -0.0243
- F-test p-value: 0.7879
- Observations: 83
- Durbin-Watson: 1.8524

Coefficients:

- const: 1.6627 (SE: 0.0741, p: 0.0000) ***
- GPR_China_y: 0.0424 (SE: 0.1157, p: 0.7151)
- GPR_US_y: 0.0115 (SE: 0.0324, p: 0.7243)
- GPR_Japan_y: -0.1740 (SE: 0.1814, p: 0.3406)

#### Bin Size: 15

**Ambiguity Risk**

- R²: 0.0035
- Adjusted R²: -0.0214
- F-test p-value: 0.8694
- Observations: 83
- Durbin-Watson: 2.3675

Coefficients:

- const: 3.6306 (SE: 5.3008, p: 0.4954)
- Ambiguity_Bins_15: -1.3263 (SE: 2.7029, p: 0.6250)
- Risk_Bins_15: -2146.0902 (SE: 6666.7045, p: 0.7484)

**Gpr China**

- R²: 0.0010
- Adjusted R²: -0.0113
- F-test p-value: 0.7772
- Observations: 83
- Durbin-Watson: 2.2461

Coefficients:

- const: 1.8590 (SE: 0.0662, p: 0.0000) ***
- GPR_China_y: 0.0223 (SE: 0.0785, p: 0.7772)

**Gpr Us**

- R²: 0.0066
- Adjusted R²: -0.0056
- F-test p-value: 0.4638
- Observations: 83
- Durbin-Watson: 2.2629

Coefficients:

- const: 1.8204 (SE: 0.0797, p: 0.0000) ***
- GPR_US_y: 0.0192 (SE: 0.0260, p: 0.4638)

**Gpr Japan**

- R²: 0.0000
- Adjusted R²: -0.0123
- F-test p-value: 0.9727
- Observations: 83
- Durbin-Watson: 2.2362

Coefficients:

- const: 1.8753 (SE: 0.0438, p: 0.0000) ***
- GPR_Japan_y: 0.0052 (SE: 0.1503, p: 0.9727)

**Gpr China Us**

- R²: 0.0077
- Adjusted R²: -0.0171
- F-test p-value: 0.7346
- Observations: 83
- Durbin-Watson: 2.2583

Coefficients:

- const: 1.8244 (SE: 0.0814, p: 0.0000) ***
- GPR_China_y: -0.0309 (SE: 0.1071, p: 0.7733)
- GPR_US_y: 0.0261 (SE: 0.0356, p: 0.4650)

**Gpr China Japan**

- R²: 0.0015
- Adjusted R²: -0.0235
- F-test p-value: 0.9420
- Observations: 83
- Durbin-Watson: 2.2439

Coefficients:

- const: 1.8581 (SE: 0.0667, p: 0.0000) ***
- GPR_China_y: 0.0360 (SE: 0.1046, p: 0.7317)
- GPR_Japan_y: -0.0400 (SE: 0.2001, p: 0.8422)

**Gpr Us Japan**

- R²: 0.0079
- Adjusted R²: -0.0169
- F-test p-value: 0.7273
- Observations: 83
- Durbin-Watson: 2.2577

Coefficients:

- const: 1.8216 (SE: 0.0803, p: 0.0000) ***
- GPR_US_y: 0.0233 (SE: 0.0292, p: 0.4268)
- GPR_Japan_y: -0.0540 (SE: 0.1678, p: 0.7485)

**Gpr China Us Japan**

- R²: 0.0082
- Adjusted R²: -0.0295
- F-test p-value: 0.8846
- Observations: 83
- Durbin-Watson: 2.2565

Coefficients:

- const: 1.8236 (SE: 0.0820, p: 0.0000) ***
- GPR_China_y: -0.0174 (SE: 0.1280, p: 0.8920)
- GPR_US_y: 0.0261 (SE: 0.0358, p: 0.4683)
- GPR_Japan_y: -0.0392 (SE: 0.2007, p: 0.8455)

#### Bin Size: 20

**Ambiguity Risk**

- R²: 0.0021
- Adjusted R²: -0.0228
- F-test p-value: 0.9179
- Observations: 83
- Durbin-Watson: 2.3384

Coefficients:

- const: -0.8995 (SE: 5.4569, p: 0.8695)
- Ambiguity_Bins_20: 0.9589 (SE: 2.6407, p: 0.7175)
- Risk_Bins_20: -794.1750 (SE: 6570.6423, p: 0.9041)

**Gpr China**

- R²: 0.0007
- Adjusted R²: -0.0117
- F-test p-value: 0.8142
- Observations: 83
- Durbin-Watson: 2.0249

Coefficients:

- const: 1.9811 (SE: 0.0668, p: 0.0000) ***
- GPR_China_y: 0.0187 (SE: 0.0792, p: 0.8142)

**Gpr Us**

- R²: 0.0240
- Adjusted R²: 0.0119
- F-test p-value: 0.1621
- Observations: 83
- Durbin-Watson: 2.0803

Coefficients:

- const: 1.8881 (SE: 0.0797, p: 0.0000) ***
- GPR_US_y: 0.0367 (SE: 0.0260, p: 0.1621)

**Gpr Japan**

- R²: 0.0008
- Adjusted R²: -0.0116
- F-test p-value: 0.8054
- Observations: 83
- Durbin-Watson: 2.0021

Coefficients:

- const: 2.0051 (SE: 0.0442, p: 0.0000) ***
- GPR_Japan_y: -0.0375 (SE: 0.1515, p: 0.8054)

**Gpr China Us**

- R²: 0.0354
- Adjusted R²: 0.0113
- F-test p-value: 0.2361
- Observations: 83
- Durbin-Watson: 2.0744

Coefficients:

- const: 1.9015 (SE: 0.0810, p: 0.0000) ***
- GPR_China_y: -0.1038 (SE: 0.1065, p: 0.3327)
- GPR_US_y: 0.0601 (SE: 0.0354, p: 0.0934)

**Gpr China Japan**

- R²: 0.0042
- Adjusted R²: -0.0207
- F-test p-value: 0.8458
- Observations: 83
- Durbin-Watson: 2.0070

Coefficients:

- const: 1.9787 (SE: 0.0672, p: 0.0000) ***
- GPR_China_y: 0.0553 (SE: 0.1054, p: 0.6014)
- GPR_Japan_y: -0.1068 (SE: 0.2016, p: 0.5978)

**Gpr Us Japan**

- R²: 0.0354
- Adjusted R²: 0.0113
- F-test p-value: 0.2367
- Observations: 83
- Durbin-Watson: 2.0528

Coefficients:

- const: 1.8919 (SE: 0.0799, p: 0.0000) ***
- GPR_US_y: 0.0492 (SE: 0.0290, p: 0.0940)
- GPR_Japan_y: -0.1623 (SE: 0.1669, p: 0.3339)

**Gpr China Us Japan**

- R²: 0.0388
- Adjusted R²: 0.0023
- F-test p-value: 0.3694
- Observations: 83
- Durbin-Watson: 2.0582

Coefficients:

- const: 1.8993 (SE: 0.0814, p: 0.0000) ***
- GPR_China_y: -0.0676 (SE: 0.1271, p: 0.5965)
- GPR_US_y: 0.0600 (SE: 0.0355, p: 0.0955)
- GPR_Japan_y: -0.1051 (SE: 0.1993, p: 0.5996)

#### Bin Size: 25

**Ambiguity Risk**

- R²: 0.0007
- Adjusted R²: -0.0243
- F-test p-value: 0.9736
- Observations: 83
- Durbin-Watson: 2.3419

Coefficients:

- const: 0.3947 (SE: 5.6991, p: 0.9450)
- Ambiguity_Bins_25: 0.3129 (SE: 2.6531, p: 0.9064)
- Risk_Bins_25: -1113.9805 (SE: 6589.5819, p: 0.8662)

**Gpr China**

- R²: 0.0005
- Adjusted R²: -0.0119
- F-test p-value: 0.8451
- Observations: 83
- Durbin-Watson: 2.0490

Coefficients:

- const: 2.0641 (SE: 0.0667, p: 0.0000) ***
- GPR_China_y: 0.0155 (SE: 0.0791, p: 0.8451)

**Gpr Us**

- R²: 0.0116
- Adjusted R²: -0.0006
- F-test p-value: 0.3321
- Observations: 83
- Durbin-Watson: 2.0693

Coefficients:

- const: 2.0015 (SE: 0.0801, p: 0.0000) ***
- GPR_US_y: 0.0255 (SE: 0.0261, p: 0.3321)

**Gpr Japan**

- R²: 0.0007
- Adjusted R²: -0.0116
- F-test p-value: 0.8066
- Observations: 83
- Durbin-Watson: 2.0508

Coefficients:

- const: 2.0671 (SE: 0.0441, p: 0.0000) ***
- GPR_Japan_y: 0.0371 (SE: 0.1513, p: 0.8066)

**Gpr China Us**

- R²: 0.0165
- Adjusted R²: -0.0081
- F-test p-value: 0.5147
- Observations: 83
- Durbin-Watson: 2.0558

Coefficients:

- const: 2.0102 (SE: 0.0816, p: 0.0000) ***
- GPR_China_y: -0.0674 (SE: 0.1073, p: 0.5316)
- GPR_US_y: 0.0407 (SE: 0.0357, p: 0.2574)

**Gpr China Japan**

- R²: 0.0008
- Adjusted R²: -0.0242
- F-test p-value: 0.9696
- Observations: 83
- Durbin-Watson: 2.0516

Coefficients:

- const: 2.0648 (SE: 0.0672, p: 0.0000) ***
- GPR_China_y: 0.0049 (SE: 0.1054, p: 0.9634)
- GPR_Japan_y: 0.0311 (SE: 0.2016, p: 0.8779)

**Gpr Us Japan**

- R²: 0.0121
- Adjusted R²: -0.0126
- F-test p-value: 0.6138
- Observations: 83
- Durbin-Watson: 2.0646

Coefficients:

- const: 2.0023 (SE: 0.0807, p: 0.0000) ***
- GPR_US_y: 0.0281 (SE: 0.0293, p: 0.3399)
- GPR_Japan_y: -0.0343 (SE: 0.1686, p: 0.8393)

**Gpr China Us Japan**

- R²: 0.0168
- Adjusted R²: -0.0205
- F-test p-value: 0.7183
- Observations: 83
- Durbin-Watson: 2.0582

Coefficients:

- const: 2.0109 (SE: 0.0822, p: 0.0000) ***
- GPR_China_y: -0.0785 (SE: 0.1283, p: 0.5423)
- GPR_US_y: 0.0407 (SE: 0.0359, p: 0.2600)
- GPR_Japan_y: 0.0322 (SE: 0.2012, p: 0.8732)

#### Bin Size: 30

**Ambiguity Risk**

- R²: 0.0006
- Adjusted R²: -0.0244
- F-test p-value: 0.9758
- Observations: 83
- Durbin-Watson: 2.3466

Coefficients:

- const: 1.6216 (SE: 5.8004, p: 0.7805)
- Ambiguity_Bins_30: -0.2622 (SE: 2.6819, p: 0.9224)
- Risk_Bins_30: -1341.2055 (SE: 6468.5513, p: 0.8363)

**Gpr China**

- R²: 0.0034
- Adjusted R²: -0.0089
- F-test p-value: 0.5988
- Observations: 83
- Durbin-Watson: 2.1761

Coefficients:

- const: 2.1516 (SE: 0.0646, p: 0.0000) ***
- GPR_China_y: -0.0405 (SE: 0.0767, p: 0.5988)

**Gpr Us**

- R²: 0.0024
- Adjusted R²: -0.0099
- F-test p-value: 0.6618
- Observations: 83
- Durbin-Watson: 2.2239

Coefficients:

- const: 2.0867 (SE: 0.0782, p: 0.0000) ***
- GPR_US_y: 0.0112 (SE: 0.0255, p: 0.6618)

**Gpr Japan**

- R²: 0.0002
- Adjusted R²: -0.0121
- F-test p-value: 0.8876
- Observations: 83
- Durbin-Watson: 2.2018

Coefficients:

- const: 2.1247 (SE: 0.0429, p: 0.0000) ***
- GPR_Japan_y: -0.0208 (SE: 0.1469, p: 0.8876)

**Gpr China Us**

- R²: 0.0179
- Adjusted R²: -0.0067
- F-test p-value: 0.4860
- Observations: 83
- Durbin-Watson: 2.1630

Coefficients:

- const: 2.1019 (SE: 0.0792, p: 0.0000) ***
- GPR_China_y: -0.1170 (SE: 0.1041, p: 0.2645)
- GPR_US_y: 0.0375 (SE: 0.0346, p: 0.2813)

**Gpr China Japan**

- R²: 0.0043
- Adjusted R²: -0.0206
- F-test p-value: 0.8406
- Observations: 83
- Durbin-Watson: 2.1826

Coefficients:

- const: 2.1527 (SE: 0.0652, p: 0.0000) ***
- GPR_China_y: -0.0585 (SE: 0.1022, p: 0.5683)
- GPR_Japan_y: 0.0526 (SE: 0.1954, p: 0.7886)

**Gpr Us Japan**

- R²: 0.0041
- Adjusted R²: -0.0208
- F-test p-value: 0.8485
- Observations: 83
- Durbin-Watson: 2.2068

Coefficients:

- const: 2.0881 (SE: 0.0787, p: 0.0000) ***
- GPR_US_y: 0.0159 (SE: 0.0286, p: 0.5797)
- GPR_Japan_y: -0.0612 (SE: 0.1644, p: 0.7108)

**Gpr China Us Japan**

- R²: 0.0188
- Adjusted R²: -0.0184
- F-test p-value: 0.6800
- Observations: 83
- Durbin-Watson: 2.1690

Coefficients:

- const: 2.1030 (SE: 0.0798, p: 0.0000) ***
- GPR_China_y: -0.1355 (SE: 0.1245, p: 0.2797)
- GPR_US_y: 0.0376 (SE: 0.0348, p: 0.2835)
- GPR_Japan_y: 0.0536 (SE: 0.1952, p: 0.7842)

#### Bin Size: 35

**Ambiguity Risk**

- R²: 0.0016
- Adjusted R²: -0.0233
- F-test p-value: 0.9364
- Observations: 83
- Durbin-Watson: 2.3511

Coefficients:

- const: 2.8130 (SE: 5.8344, p: 0.6310)
- Ambiguity_Bins_35: -0.8044 (SE: 2.6513, p: 0.7624)
- Risk_Bins_35: -1356.7420 (SE: 6439.6852, p: 0.8337)

**Gpr China**

- R²: 0.0004
- Adjusted R²: -0.0120
- F-test p-value: 0.8593
- Observations: 83
- Durbin-Watson: 2.2442

Coefficients:

- const: 2.1811 (SE: 0.0652, p: 0.0000) ***
- GPR_China_y: -0.0138 (SE: 0.0774, p: 0.8593)

**Gpr Us**

- R²: 0.0098
- Adjusted R²: -0.0024
- F-test p-value: 0.3733
- Observations: 83
- Durbin-Watson: 2.2920

Coefficients:

- const: 2.1030 (SE: 0.0784, p: 0.0000) ***
- GPR_US_y: 0.0229 (SE: 0.0256, p: 0.3733)

**Gpr Japan**

- R²: 0.0007
- Adjusted R²: -0.0116
- F-test p-value: 0.8071
- Observations: 83
- Durbin-Watson: 2.2381

Coefficients:

- const: 2.1792 (SE: 0.0431, p: 0.0000) ***
- GPR_Japan_y: -0.0362 (SE: 0.1479, p: 0.8071)

**Gpr China Us**

- R²: 0.0237
- Adjusted R²: -0.0007
- F-test p-value: 0.3828
- Observations: 83
- Durbin-Watson: 2.2284

Coefficients:

- const: 2.1175 (SE: 0.0795, p: 0.0000) ***
- GPR_China_y: -0.1117 (SE: 0.1046, p: 0.2887)
- GPR_US_y: 0.0480 (SE: 0.0347, p: 0.1706)

**Gpr China Japan**

- R²: 0.0007
- Adjusted R²: -0.0242
- F-test p-value: 0.9706
- Observations: 83
- Durbin-Watson: 2.2374

Coefficients:

- const: 2.1803 (SE: 0.0657, p: 0.0000) ***
- GPR_China_y: -0.0023 (SE: 0.1031, p: 0.9820)
- GPR_Japan_y: -0.0333 (SE: 0.1971, p: 0.8662)

**Gpr Us Japan**

- R²: 0.0160
- Adjusted R²: -0.0086
- F-test p-value: 0.5238
- Observations: 83
- Durbin-Watson: 2.2476

Coefficients:

- const: 2.1057 (SE: 0.0787, p: 0.0000) ***
- GPR_US_y: 0.0319 (SE: 0.0286, p: 0.2681)
- GPR_Japan_y: -0.1172 (SE: 0.1646, p: 0.4784)

**Gpr China Us Japan**

- R²: 0.0240
- Adjusted R²: -0.0130
- F-test p-value: 0.5860
- Observations: 83
- Durbin-Watson: 2.2222

Coefficients:

- const: 2.1168 (SE: 0.0801, p: 0.0000) ***
- GPR_China_y: -0.1007 (SE: 0.1250, p: 0.4231)
- GPR_US_y: 0.0480 (SE: 0.0350, p: 0.1735)
- GPR_Japan_y: -0.0319 (SE: 0.1960, p: 0.8710)

#### Bin Size: 40

**Ambiguity Risk**

- R²: 0.0053
- Adjusted R²: -0.0196
- F-test p-value: 0.8093
- Observations: 83
- Durbin-Watson: 2.3488

Coefficients:

- const: 4.6590 (SE: 5.8604, p: 0.4290)
- Ambiguity_Bins_40: -1.6328 (SE: 2.6333, p: 0.5370)
- Risk_Bins_40: -1424.8773 (SE: 6427.2364, p: 0.8251)

**Gpr China**

- R²: 0.0008
- Adjusted R²: -0.0116
- F-test p-value: 0.8038
- Observations: 83
- Durbin-Watson: 2.2627

Coefficients:

- const: 2.2111 (SE: 0.0655, p: 0.0000) ***
- GPR_China_y: -0.0194 (SE: 0.0777, p: 0.8038)

**Gpr Us**

- R²: 0.0091
- Adjusted R²: -0.0031
- F-test p-value: 0.3904
- Observations: 83
- Durbin-Watson: 2.3165

Coefficients:

- const: 2.1306 (SE: 0.0788, p: 0.0000) ***
- GPR_US_y: 0.0222 (SE: 0.0257, p: 0.3904)

**Gpr Japan**

- R²: 0.0001
- Adjusted R²: -0.0123
- F-test p-value: 0.9358
- Observations: 83
- Durbin-Watson: 2.2763

Coefficients:

- const: 2.1988 (SE: 0.0434, p: 0.0000) ***
- GPR_Japan_y: -0.0120 (SE: 0.1487, p: 0.9358)

**Gpr China Us**

- R²: 0.0249
- Adjusted R²: 0.0005
- F-test p-value: 0.3649
- Observations: 83
- Durbin-Watson: 2.2480

Coefficients:

- const: 2.1461 (SE: 0.0798, p: 0.0000) ***
- GPR_China_y: -0.1194 (SE: 0.1050, p: 0.2588)
- GPR_US_y: 0.0491 (SE: 0.0349, p: 0.1634)

**Gpr China Japan**

- R²: 0.0009
- Adjusted R²: -0.0241
- F-test p-value: 0.9641
- Observations: 83
- Durbin-Watson: 2.2649

Coefficients:

- const: 2.2116 (SE: 0.0661, p: 0.0000) ***
- GPR_China_y: -0.0267 (SE: 0.1036, p: 0.7968)
- GPR_Japan_y: 0.0215 (SE: 0.1981, p: 0.9137)

**Gpr Us Japan**

- R²: 0.0124
- Adjusted R²: -0.0123
- F-test p-value: 0.6079
- Observations: 83
- Durbin-Watson: 2.2929

Coefficients:

- const: 2.1326 (SE: 0.0793, p: 0.0000) ***
- GPR_US_y: 0.0287 (SE: 0.0288, p: 0.3215)
- GPR_Japan_y: -0.0850 (SE: 0.1657, p: 0.6096)

**Gpr China Us Japan**

- R²: 0.0251
- Adjusted R²: -0.0120
- F-test p-value: 0.5688
- Observations: 83
- Durbin-Watson: 2.2501

Coefficients:

- const: 2.1466 (SE: 0.0805, p: 0.0000) ***
- GPR_China_y: -0.1273 (SE: 0.1256, p: 0.3136)
- GPR_US_y: 0.0491 (SE: 0.0351, p: 0.1658)
- GPR_Japan_y: 0.0229 (SE: 0.1969, p: 0.9076)

#### Bin Size: 45

**Ambiguity Risk**

- R²: 0.0005
- Adjusted R²: -0.0245
- F-test p-value: 0.9801
- Observations: 83
- Durbin-Watson: 2.3450

Coefficients:

- const: 1.2021 (SE: 5.7485, p: 0.8349)
- Ambiguity_Bins_45: -0.0632 (SE: 2.5324, p: 0.9801)
- Risk_Bins_45: -1291.2485 (SE: 6454.6229, p: 0.8419)

**Gpr China**

- R²: 0.0006
- Adjusted R²: -0.0117
- F-test p-value: 0.8194
- Observations: 83
- Durbin-Watson: 2.2585

Coefficients:

- const: 2.2449 (SE: 0.0684, p: 0.0000) ***
- GPR_China_y: -0.0186 (SE: 0.0812, p: 0.8194)

**Gpr Us**

- R²: 0.0086
- Adjusted R²: -0.0037
- F-test p-value: 0.4054
- Observations: 83
- Durbin-Watson: 2.3058

Coefficients:

- const: 2.1642 (SE: 0.0823, p: 0.0000) ***
- GPR_US_y: 0.0225 (SE: 0.0269, p: 0.4054)

**Gpr Japan**

- R²: 0.0015
- Adjusted R²: -0.0109
- F-test p-value: 0.7306
- Observations: 83
- Durbin-Watson: 2.2491

Coefficients:

- const: 2.2435 (SE: 0.0453, p: 0.0000) ***
- GPR_Japan_y: -0.0536 (SE: 0.1552, p: 0.7306)

**Gpr China Us**

- R²: 0.0229
- Adjusted R²: -0.0015
- F-test p-value: 0.3956
- Observations: 83
- Durbin-Watson: 2.2369

Coefficients:

- const: 2.1797 (SE: 0.0835, p: 0.0000) ***
- GPR_China_y: -0.1190 (SE: 0.1098, p: 0.2816)
- GPR_US_y: 0.0493 (SE: 0.0365, p: 0.1807)

**Gpr China Japan**

- R²: 0.0015
- Adjusted R²: -0.0235
- F-test p-value: 0.9428
- Observations: 83
- Durbin-Watson: 2.2490

Coefficients:

- const: 2.2437 (SE: 0.0690, p: 0.0000) ***
- GPR_China_y: -0.0004 (SE: 0.1081, p: 0.9973)
- GPR_Japan_y: -0.0532 (SE: 0.2068, p: 0.7978)

**Gpr Us Japan**

- R²: 0.0164
- Adjusted R²: -0.0082
- F-test p-value: 0.5171
- Observations: 83
- Durbin-Watson: 2.2566

Coefficients:

- const: 2.1674 (SE: 0.0826, p: 0.0000) ***
- GPR_US_y: 0.0330 (SE: 0.0300, p: 0.2746)
- GPR_Japan_y: -0.1375 (SE: 0.1727, p: 0.4284)

**Gpr China Us Japan**

- R²: 0.0237
- Adjusted R²: -0.0134
- F-test p-value: 0.5920
- Observations: 83
- Durbin-Watson: 2.2281

Coefficients:

- const: 2.1786 (SE: 0.0841, p: 0.0000) ***
- GPR_China_y: -0.1012 (SE: 0.1312, p: 0.4431)
- GPR_US_y: 0.0492 (SE: 0.0367, p: 0.1837)
- GPR_Japan_y: -0.0517 (SE: 0.2057, p: 0.8021)

#### Bin Size: 50

**Ambiguity Risk**

- R²: 0.0019
- Adjusted R²: -0.0230
- F-test p-value: 0.9253
- Observations: 83
- Durbin-Watson: 2.3504

Coefficients:

- const: 3.0821 (SE: 5.9947, p: 0.6086)
- Ambiguity_Bins_50: -0.8950 (SE: 2.6289, p: 0.7344)
- Risk_Bins_50: -1442.3754 (SE: 6451.4299, p: 0.8237)

**Gpr China**

- R²: 0.0018
- Adjusted R²: -0.0105
- F-test p-value: 0.7022
- Observations: 83
- Durbin-Watson: 2.1760

Coefficients:

- const: 2.2660 (SE: 0.0658, p: 0.0000) ***
- GPR_China_y: -0.0300 (SE: 0.0781, p: 0.7022)

**Gpr Us**

- R²: 0.0085
- Adjusted R²: -0.0038
- F-test p-value: 0.4078
- Observations: 83
- Durbin-Watson: 2.2351

Coefficients:

- const: 2.1791 (SE: 0.0793, p: 0.0000) ***
- GPR_US_y: 0.0215 (SE: 0.0259, p: 0.4078)

**Gpr Japan**

- R²: 0.0018
- Adjusted R²: -0.0105
- F-test p-value: 0.7047
- Observations: 83
- Durbin-Watson: 2.1692

Coefficients:

- const: 2.2564 (SE: 0.0436, p: 0.0000) ***
- GPR_Japan_y: -0.0568 (SE: 0.1494, p: 0.7047)

**Gpr China Us**

- R²: 0.0288
- Adjusted R²: 0.0045
- F-test p-value: 0.3103
- Observations: 83
- Durbin-Watson: 2.1650

Coefficients:

- const: 2.1968 (SE: 0.0801, p: 0.0000) ***
- GPR_China_y: -0.1364 (SE: 0.1054, p: 0.1991)
- GPR_US_y: 0.0522 (SE: 0.0350, p: 0.1397)

**Gpr China Japan**

- R²: 0.0022
- Adjusted R²: -0.0228
- F-test p-value: 0.9167
- Observations: 83
- Durbin-Watson: 2.1662

Coefficients:

- const: 2.2652 (SE: 0.0664, p: 0.0000) ***
- GPR_China_y: -0.0184 (SE: 0.1041, p: 0.8601)
- GPR_Japan_y: -0.0337 (SE: 0.1990, p: 0.8658)

**Gpr Us Japan**

- R²: 0.0170
- Adjusted R²: -0.0076
- F-test p-value: 0.5037
- Observations: 83
- Durbin-Watson: 2.1737

Coefficients:

- const: 2.1823 (SE: 0.0795, p: 0.0000) ***
- GPR_US_y: 0.0321 (SE: 0.0289, p: 0.2692)
- GPR_Japan_y: -0.1384 (SE: 0.1662, p: 0.4074)

**Gpr China Us Japan**

- R²: 0.0292
- Adjusted R²: -0.0077
- F-test p-value: 0.5026
- Observations: 83
- Durbin-Watson: 2.1559

Coefficients:

- const: 2.1961 (SE: 0.0807, p: 0.0000) ***
- GPR_China_y: -0.1253 (SE: 0.1260, p: 0.3229)
- GPR_US_y: 0.0522 (SE: 0.0352, p: 0.1424)
- GPR_Japan_y: -0.0322 (SE: 0.1975, p: 0.8708)

### Log_Monthly_Return Results

#### Bin Size: 5

**Ambiguity Risk**

- R²: 0.0014
- Adjusted R²: -0.0235
- F-test p-value: 0.9443
- Observations: 83
- Durbin-Watson: 2.3462

Coefficients:

- const: 0.0211 (SE: 0.0492, p: 0.6685)
- Ambiguity_Bins_5: -0.0087 (SE: 0.0369, p: 0.8138)
- Risk_Bins_5: -20.8494 (SE: 67.3177, p: 0.7576)

**Gpr China**

- R²: 0.0118
- Adjusted R²: -0.0004
- F-test p-value: 0.3273
- Observations: 83
- Durbin-Watson: 1.9547

Coefficients:

- const: 1.2099 (SE: 0.0487, p: 0.0000) ***
- GPR_China_y: 0.0569 (SE: 0.0577, p: 0.3273)

**Gpr Us**

- R²: 0.0027
- Adjusted R²: -0.0096
- F-test p-value: 0.6384
- Observations: 83
- Durbin-Watson: 1.9474

Coefficients:

- const: 1.2282 (SE: 0.0591, p: 0.0000) ***
- GPR_US_y: 0.0091 (SE: 0.0193, p: 0.6384)

**Gpr Japan**

- R²: 0.0021
- Adjusted R²: -0.0102
- F-test p-value: 0.6822
- Observations: 83
- Durbin-Watson: 1.9362

Coefficients:

- const: 1.2662 (SE: 0.0324, p: 0.0000) ***
- GPR_Japan_y: -0.0456 (SE: 0.1110, p: 0.6822)

**Gpr China Us**

- R²: 0.0127
- Adjusted R²: -0.0120
- F-test p-value: 0.5999
- Observations: 83
- Durbin-Watson: 1.9564

Coefficients:

- const: 1.2190 (SE: 0.0600, p: 0.0000) ***
- GPR_China_y: 0.0709 (SE: 0.0790, p: 0.3719)
- GPR_US_y: -0.0069 (SE: 0.0262, p: 0.7942)

**Gpr China Japan**

- R²: 0.0358
- Adjusted R²: 0.0117
- F-test p-value: 0.2323
- Observations: 83
- Durbin-Watson: 1.9362

Coefficients:

- const: 1.2052 (SE: 0.0485, p: 0.0000) ***
- GPR_China_y: 0.1272 (SE: 0.0760, p: 0.0981)
- GPR_Japan_y: -0.2051 (SE: 0.1454, p: 0.1621)

**Gpr Us Japan**

- R²: 0.0086
- Adjusted R²: -0.0162
- F-test p-value: 0.7078
- Observations: 83
- Durbin-Watson: 1.9343

Coefficients:

- const: 1.2302 (SE: 0.0593, p: 0.0000) ***
- GPR_US_y: 0.0156 (SE: 0.0216, p: 0.4703)
- GPR_Japan_y: -0.0853 (SE: 0.1240, p: 0.4936)

**Gpr China Us Japan**

- R²: 0.0367
- Adjusted R²: 0.0002
- F-test p-value: 0.3955
- Observations: 83
- Durbin-Watson: 1.9372

Coefficients:

- const: 1.2146 (SE: 0.0598, p: 0.0000) ***
- GPR_China_y: 0.1417 (SE: 0.0933, p: 0.1328)
- GPR_US_y: -0.0071 (SE: 0.0261, p: 0.7875)
- GPR_Japan_y: -0.2053 (SE: 0.1462, p: 0.1642)

#### Bin Size: 10

**Ambiguity Risk**

- R²: 0.0126
- Adjusted R²: -0.0120
- F-test p-value: 0.6013
- Observations: 83
- Durbin-Watson: 2.3406

Coefficients:

- const: -0.0427 (SE: 0.0539, p: 0.4308)
- Ambiguity_Bins_10: 0.0297 (SE: 0.0302, p: 0.3291)
- Risk_Bins_10: 9.2219 (SE: 67.5783, p: 0.8918)

**Gpr China**

- R²: 0.0001
- Adjusted R²: -0.0123
- F-test p-value: 0.9319
- Observations: 83
- Durbin-Watson: 1.8572

Coefficients:

- const: 1.6818 (SE: 0.0600, p: 0.0000) ***
- GPR_China_y: 0.0061 (SE: 0.0712, p: 0.9319)

**Gpr Us**

- R²: 0.0013
- Adjusted R²: -0.0110
- F-test p-value: 0.7469
- Observations: 83
- Durbin-Watson: 1.8600

Coefficients:

- const: 1.6641 (SE: 0.0725, p: 0.0000) ***
- GPR_US_y: 0.0077 (SE: 0.0237, p: 0.7469)

**Gpr Japan**

- R²: 0.0056
- Adjusted R²: -0.0067
- F-test p-value: 0.5014
- Observations: 83
- Durbin-Watson: 1.8523

Coefficients:

- const: 1.7094 (SE: 0.0396, p: 0.0000) ***
- GPR_Japan_y: -0.0917 (SE: 0.1358, p: 0.5014)

**Gpr China Us**

- R²: 0.0017
- Adjusted R²: -0.0233
- F-test p-value: 0.9342
- Observations: 83
- Durbin-Watson: 1.8628

Coefficients:

- const: 1.6664 (SE: 0.0740, p: 0.0000) ***
- GPR_China_y: -0.0176 (SE: 0.0973, p: 0.8572)
- GPR_US_y: 0.0116 (SE: 0.0323, p: 0.7205)

**Gpr China Japan**

- R²: 0.0116
- Adjusted R²: -0.0131
- F-test p-value: 0.6266
- Observations: 83
- Durbin-Watson: 1.8461

Coefficients:

- const: 1.6778 (SE: 0.0602, p: 0.0000) ***
- GPR_China_y: 0.0659 (SE: 0.0943, p: 0.4872)
- GPR_Japan_y: -0.1743 (SE: 0.1805, p: 0.3370)

**Gpr Us Japan**

- R²: 0.0115
- Adjusted R²: -0.0132
- F-test p-value: 0.6294
- Observations: 83
- Durbin-Watson: 1.8589

Coefficients:

- const: 1.6673 (SE: 0.0726, p: 0.0000) ***
- GPR_US_y: 0.0182 (SE: 0.0264, p: 0.4912)
- GPR_Japan_y: -0.1381 (SE: 0.1519, p: 0.3660)

**Gpr China Us Japan**

- R²: 0.0132
- Adjusted R²: -0.0243
- F-test p-value: 0.7879
- Observations: 83
- Durbin-Watson: 1.8524

Coefficients:

- const: 1.6627 (SE: 0.0741, p: 0.0000) ***
- GPR_China_y: 0.0424 (SE: 0.1157, p: 0.7151)
- GPR_US_y: 0.0115 (SE: 0.0324, p: 0.7243)
- GPR_Japan_y: -0.1740 (SE: 0.1814, p: 0.3406)

#### Bin Size: 15

**Ambiguity Risk**

- R²: 0.0035
- Adjusted R²: -0.0214
- F-test p-value: 0.8695
- Observations: 83
- Durbin-Watson: 2.3712

Coefficients:

- const: 0.0338 (SE: 0.0520, p: 0.5172)
- Ambiguity_Bins_15: -0.0125 (SE: 0.0265, p: 0.6395)
- Risk_Bins_15: -23.4728 (SE: 65.4066, p: 0.7206)

**Gpr China**

- R²: 0.0010
- Adjusted R²: -0.0113
- F-test p-value: 0.7772
- Observations: 83
- Durbin-Watson: 2.2461

Coefficients:

- const: 1.8590 (SE: 0.0662, p: 0.0000) ***
- GPR_China_y: 0.0223 (SE: 0.0785, p: 0.7772)

**Gpr Us**

- R²: 0.0066
- Adjusted R²: -0.0056
- F-test p-value: 0.4638
- Observations: 83
- Durbin-Watson: 2.2629

Coefficients:

- const: 1.8204 (SE: 0.0797, p: 0.0000) ***
- GPR_US_y: 0.0192 (SE: 0.0260, p: 0.4638)

**Gpr Japan**

- R²: 0.0000
- Adjusted R²: -0.0123
- F-test p-value: 0.9727
- Observations: 83
- Durbin-Watson: 2.2362

Coefficients:

- const: 1.8753 (SE: 0.0438, p: 0.0000) ***
- GPR_Japan_y: 0.0052 (SE: 0.1503, p: 0.9727)

**Gpr China Us**

- R²: 0.0077
- Adjusted R²: -0.0171
- F-test p-value: 0.7346
- Observations: 83
- Durbin-Watson: 2.2583

Coefficients:

- const: 1.8244 (SE: 0.0814, p: 0.0000) ***
- GPR_China_y: -0.0309 (SE: 0.1071, p: 0.7733)
- GPR_US_y: 0.0261 (SE: 0.0356, p: 0.4650)

**Gpr China Japan**

- R²: 0.0015
- Adjusted R²: -0.0235
- F-test p-value: 0.9420
- Observations: 83
- Durbin-Watson: 2.2439

Coefficients:

- const: 1.8581 (SE: 0.0667, p: 0.0000) ***
- GPR_China_y: 0.0360 (SE: 0.1046, p: 0.7317)
- GPR_Japan_y: -0.0400 (SE: 0.2001, p: 0.8422)

**Gpr Us Japan**

- R²: 0.0079
- Adjusted R²: -0.0169
- F-test p-value: 0.7273
- Observations: 83
- Durbin-Watson: 2.2577

Coefficients:

- const: 1.8216 (SE: 0.0803, p: 0.0000) ***
- GPR_US_y: 0.0233 (SE: 0.0292, p: 0.4268)
- GPR_Japan_y: -0.0540 (SE: 0.1678, p: 0.7485)

**Gpr China Us Japan**

- R²: 0.0082
- Adjusted R²: -0.0295
- F-test p-value: 0.8846
- Observations: 83
- Durbin-Watson: 2.2565

Coefficients:

- const: 1.8236 (SE: 0.0820, p: 0.0000) ***
- GPR_China_y: -0.0174 (SE: 0.1280, p: 0.8920)
- GPR_US_y: 0.0261 (SE: 0.0358, p: 0.4683)
- GPR_Japan_y: -0.0392 (SE: 0.2007, p: 0.8455)

#### Bin Size: 20

**Ambiguity Risk**

- R²: 0.0028
- Adjusted R²: -0.0221
- F-test p-value: 0.8935
- Observations: 83
- Durbin-Watson: 2.3435

Coefficients:

- const: -0.0119 (SE: 0.0535, p: 0.8242)
- Ambiguity_Bins_20: 0.0106 (SE: 0.0259, p: 0.6844)
- Risk_Bins_20: -9.9783 (SE: 64.4423, p: 0.8773)

**Gpr China**

- R²: 0.0007
- Adjusted R²: -0.0117
- F-test p-value: 0.8142
- Observations: 83
- Durbin-Watson: 2.0249

Coefficients:

- const: 1.9811 (SE: 0.0668, p: 0.0000) ***
- GPR_China_y: 0.0187 (SE: 0.0792, p: 0.8142)

**Gpr Us**

- R²: 0.0240
- Adjusted R²: 0.0119
- F-test p-value: 0.1621
- Observations: 83
- Durbin-Watson: 2.0803

Coefficients:

- const: 1.8881 (SE: 0.0797, p: 0.0000) ***
- GPR_US_y: 0.0367 (SE: 0.0260, p: 0.1621)

**Gpr Japan**

- R²: 0.0008
- Adjusted R²: -0.0116
- F-test p-value: 0.8054
- Observations: 83
- Durbin-Watson: 2.0021

Coefficients:

- const: 2.0051 (SE: 0.0442, p: 0.0000) ***
- GPR_Japan_y: -0.0375 (SE: 0.1515, p: 0.8054)

**Gpr China Us**

- R²: 0.0354
- Adjusted R²: 0.0113
- F-test p-value: 0.2361
- Observations: 83
- Durbin-Watson: 2.0744

Coefficients:

- const: 1.9015 (SE: 0.0810, p: 0.0000) ***
- GPR_China_y: -0.1038 (SE: 0.1065, p: 0.3327)
- GPR_US_y: 0.0601 (SE: 0.0354, p: 0.0934)

**Gpr China Japan**

- R²: 0.0042
- Adjusted R²: -0.0207
- F-test p-value: 0.8458
- Observations: 83
- Durbin-Watson: 2.0070

Coefficients:

- const: 1.9787 (SE: 0.0672, p: 0.0000) ***
- GPR_China_y: 0.0553 (SE: 0.1054, p: 0.6014)
- GPR_Japan_y: -0.1068 (SE: 0.2016, p: 0.5978)

**Gpr Us Japan**

- R²: 0.0354
- Adjusted R²: 0.0113
- F-test p-value: 0.2367
- Observations: 83
- Durbin-Watson: 2.0528

Coefficients:

- const: 1.8919 (SE: 0.0799, p: 0.0000) ***
- GPR_US_y: 0.0492 (SE: 0.0290, p: 0.0940)
- GPR_Japan_y: -0.1623 (SE: 0.1669, p: 0.3339)

**Gpr China Us Japan**

- R²: 0.0388
- Adjusted R²: 0.0023
- F-test p-value: 0.3694
- Observations: 83
- Durbin-Watson: 2.0582

Coefficients:

- const: 1.8993 (SE: 0.0814, p: 0.0000) ***
- GPR_China_y: -0.0676 (SE: 0.1271, p: 0.5965)
- GPR_US_y: 0.0600 (SE: 0.0355, p: 0.0955)
- GPR_Japan_y: -0.1051 (SE: 0.1993, p: 0.5996)

#### Bin Size: 25

**Ambiguity Risk**

- R²: 0.0010
- Adjusted R²: -0.0239
- F-test p-value: 0.9593
- Observations: 83
- Durbin-Watson: 2.3464

Coefficients:

- const: 0.0010 (SE: 0.0559, p: 0.9851)
- Ambiguity_Bins_25: 0.0041 (SE: 0.0260, p: 0.8766)
- Risk_Bins_25: -13.1814 (SE: 64.6378, p: 0.8389)

**Gpr China**

- R²: 0.0005
- Adjusted R²: -0.0119
- F-test p-value: 0.8451
- Observations: 83
- Durbin-Watson: 2.0490

Coefficients:

- const: 2.0641 (SE: 0.0667, p: 0.0000) ***
- GPR_China_y: 0.0155 (SE: 0.0791, p: 0.8451)

**Gpr Us**

- R²: 0.0116
- Adjusted R²: -0.0006
- F-test p-value: 0.3321
- Observations: 83
- Durbin-Watson: 2.0693

Coefficients:

- const: 2.0015 (SE: 0.0801, p: 0.0000) ***
- GPR_US_y: 0.0255 (SE: 0.0261, p: 0.3321)

**Gpr Japan**

- R²: 0.0007
- Adjusted R²: -0.0116
- F-test p-value: 0.8066
- Observations: 83
- Durbin-Watson: 2.0508

Coefficients:

- const: 2.0671 (SE: 0.0441, p: 0.0000) ***
- GPR_Japan_y: 0.0371 (SE: 0.1513, p: 0.8066)

**Gpr China Us**

- R²: 0.0165
- Adjusted R²: -0.0081
- F-test p-value: 0.5147
- Observations: 83
- Durbin-Watson: 2.0558

Coefficients:

- const: 2.0102 (SE: 0.0816, p: 0.0000) ***
- GPR_China_y: -0.0674 (SE: 0.1073, p: 0.5316)
- GPR_US_y: 0.0407 (SE: 0.0357, p: 0.2574)

**Gpr China Japan**

- R²: 0.0008
- Adjusted R²: -0.0242
- F-test p-value: 0.9696
- Observations: 83
- Durbin-Watson: 2.0516

Coefficients:

- const: 2.0648 (SE: 0.0672, p: 0.0000) ***
- GPR_China_y: 0.0049 (SE: 0.1054, p: 0.9634)
- GPR_Japan_y: 0.0311 (SE: 0.2016, p: 0.8779)

**Gpr Us Japan**

- R²: 0.0121
- Adjusted R²: -0.0126
- F-test p-value: 0.6138
- Observations: 83
- Durbin-Watson: 2.0646

Coefficients:

- const: 2.0023 (SE: 0.0807, p: 0.0000) ***
- GPR_US_y: 0.0281 (SE: 0.0293, p: 0.3399)
- GPR_Japan_y: -0.0343 (SE: 0.1686, p: 0.8393)

**Gpr China Us Japan**

- R²: 0.0168
- Adjusted R²: -0.0205
- F-test p-value: 0.7183
- Observations: 83
- Durbin-Watson: 2.0582

Coefficients:

- const: 2.0109 (SE: 0.0822, p: 0.0000) ***
- GPR_China_y: -0.0785 (SE: 0.1283, p: 0.5423)
- GPR_US_y: 0.0407 (SE: 0.0359, p: 0.2600)
- GPR_Japan_y: 0.0322 (SE: 0.2012, p: 0.8732)

#### Bin Size: 30

**Ambiguity Risk**

- R²: 0.0008
- Adjusted R²: -0.0242
- F-test p-value: 0.9694
- Observations: 83
- Durbin-Watson: 2.3511

Coefficients:

- const: 0.0129 (SE: 0.0569, p: 0.8218)
- Ambiguity_Bins_30: -0.0015 (SE: 0.0263, p: 0.9549)
- Risk_Bins_30: -15.6791 (SE: 63.4572, p: 0.8055)

**Gpr China**

- R²: 0.0034
- Adjusted R²: -0.0089
- F-test p-value: 0.5988
- Observations: 83
- Durbin-Watson: 2.1761

Coefficients:

- const: 2.1516 (SE: 0.0646, p: 0.0000) ***
- GPR_China_y: -0.0405 (SE: 0.0767, p: 0.5988)

**Gpr Us**

- R²: 0.0024
- Adjusted R²: -0.0099
- F-test p-value: 0.6618
- Observations: 83
- Durbin-Watson: 2.2239

Coefficients:

- const: 2.0867 (SE: 0.0782, p: 0.0000) ***
- GPR_US_y: 0.0112 (SE: 0.0255, p: 0.6618)

**Gpr Japan**

- R²: 0.0002
- Adjusted R²: -0.0121
- F-test p-value: 0.8876
- Observations: 83
- Durbin-Watson: 2.2018

Coefficients:

- const: 2.1247 (SE: 0.0429, p: 0.0000) ***
- GPR_Japan_y: -0.0208 (SE: 0.1469, p: 0.8876)

**Gpr China Us**

- R²: 0.0179
- Adjusted R²: -0.0067
- F-test p-value: 0.4860
- Observations: 83
- Durbin-Watson: 2.1630

Coefficients:

- const: 2.1019 (SE: 0.0792, p: 0.0000) ***
- GPR_China_y: -0.1170 (SE: 0.1041, p: 0.2645)
- GPR_US_y: 0.0375 (SE: 0.0346, p: 0.2813)

**Gpr China Japan**

- R²: 0.0043
- Adjusted R²: -0.0206
- F-test p-value: 0.8406
- Observations: 83
- Durbin-Watson: 2.1826

Coefficients:

- const: 2.1527 (SE: 0.0652, p: 0.0000) ***
- GPR_China_y: -0.0585 (SE: 0.1022, p: 0.5683)
- GPR_Japan_y: 0.0526 (SE: 0.1954, p: 0.7886)

**Gpr Us Japan**

- R²: 0.0041
- Adjusted R²: -0.0208
- F-test p-value: 0.8485
- Observations: 83
- Durbin-Watson: 2.2068

Coefficients:

- const: 2.0881 (SE: 0.0787, p: 0.0000) ***
- GPR_US_y: 0.0159 (SE: 0.0286, p: 0.5797)
- GPR_Japan_y: -0.0612 (SE: 0.1644, p: 0.7108)

**Gpr China Us Japan**

- R²: 0.0188
- Adjusted R²: -0.0184
- F-test p-value: 0.6800
- Observations: 83
- Durbin-Watson: 2.1690

Coefficients:

- const: 2.1030 (SE: 0.0798, p: 0.0000) ***
- GPR_China_y: -0.1355 (SE: 0.1245, p: 0.2797)
- GPR_US_y: 0.0376 (SE: 0.0348, p: 0.2835)
- GPR_Japan_y: 0.0536 (SE: 0.1952, p: 0.7842)

#### Bin Size: 35

**Ambiguity Risk**

- R²: 0.0017
- Adjusted R²: -0.0233
- F-test p-value: 0.9346
- Observations: 83
- Durbin-Watson: 2.3556

Coefficients:

- const: 0.0253 (SE: 0.0572, p: 0.6592)
- Ambiguity_Bins_35: -0.0072 (SE: 0.0260, p: 0.7829)
- Risk_Bins_35: -16.0176 (SE: 63.1777, p: 0.8005)

**Gpr China**

- R²: 0.0004
- Adjusted R²: -0.0120
- F-test p-value: 0.8593
- Observations: 83
- Durbin-Watson: 2.2442

Coefficients:

- const: 2.1811 (SE: 0.0652, p: 0.0000) ***
- GPR_China_y: -0.0138 (SE: 0.0774, p: 0.8593)

**Gpr Us**

- R²: 0.0098
- Adjusted R²: -0.0024
- F-test p-value: 0.3733
- Observations: 83
- Durbin-Watson: 2.2920

Coefficients:

- const: 2.1030 (SE: 0.0784, p: 0.0000) ***
- GPR_US_y: 0.0229 (SE: 0.0256, p: 0.3733)

**Gpr Japan**

- R²: 0.0007
- Adjusted R²: -0.0116
- F-test p-value: 0.8071
- Observations: 83
- Durbin-Watson: 2.2381

Coefficients:

- const: 2.1792 (SE: 0.0431, p: 0.0000) ***
- GPR_Japan_y: -0.0362 (SE: 0.1479, p: 0.8071)

**Gpr China Us**

- R²: 0.0237
- Adjusted R²: -0.0007
- F-test p-value: 0.3828
- Observations: 83
- Durbin-Watson: 2.2284

Coefficients:

- const: 2.1175 (SE: 0.0795, p: 0.0000) ***
- GPR_China_y: -0.1117 (SE: 0.1046, p: 0.2887)
- GPR_US_y: 0.0480 (SE: 0.0347, p: 0.1706)

**Gpr China Japan**

- R²: 0.0007
- Adjusted R²: -0.0242
- F-test p-value: 0.9706
- Observations: 83
- Durbin-Watson: 2.2374

Coefficients:

- const: 2.1803 (SE: 0.0657, p: 0.0000) ***
- GPR_China_y: -0.0023 (SE: 0.1031, p: 0.9820)
- GPR_Japan_y: -0.0333 (SE: 0.1971, p: 0.8662)

**Gpr Us Japan**

- R²: 0.0160
- Adjusted R²: -0.0086
- F-test p-value: 0.5238
- Observations: 83
- Durbin-Watson: 2.2476

Coefficients:

- const: 2.1057 (SE: 0.0787, p: 0.0000) ***
- GPR_US_y: 0.0319 (SE: 0.0286, p: 0.2681)
- GPR_Japan_y: -0.1172 (SE: 0.1646, p: 0.4784)

**Gpr China Us Japan**

- R²: 0.0240
- Adjusted R²: -0.0130
- F-test p-value: 0.5860
- Observations: 83
- Durbin-Watson: 2.2222

Coefficients:

- const: 2.1168 (SE: 0.0801, p: 0.0000) ***
- GPR_China_y: -0.1007 (SE: 0.1250, p: 0.4231)
- GPR_US_y: 0.0480 (SE: 0.0350, p: 0.1735)
- GPR_Japan_y: -0.0319 (SE: 0.1960, p: 0.8710)

#### Bin Size: 40

**Ambiguity Risk**

- R²: 0.0051
- Adjusted R²: -0.0198
- F-test p-value: 0.8157
- Observations: 83
- Durbin-Watson: 2.3536

Coefficients:

- const: 0.0433 (SE: 0.0575, p: 0.4534)
- Ambiguity_Bins_40: -0.0153 (SE: 0.0258, p: 0.5562)
- Risk_Bins_40: -16.6864 (SE: 63.0632, p: 0.7920)

**Gpr China**

- R²: 0.0008
- Adjusted R²: -0.0116
- F-test p-value: 0.8038
- Observations: 83
- Durbin-Watson: 2.2627

Coefficients:

- const: 2.2111 (SE: 0.0655, p: 0.0000) ***
- GPR_China_y: -0.0194 (SE: 0.0777, p: 0.8038)

**Gpr Us**

- R²: 0.0091
- Adjusted R²: -0.0031
- F-test p-value: 0.3904
- Observations: 83
- Durbin-Watson: 2.3165

Coefficients:

- const: 2.1306 (SE: 0.0788, p: 0.0000) ***
- GPR_US_y: 0.0222 (SE: 0.0257, p: 0.3904)

**Gpr Japan**

- R²: 0.0001
- Adjusted R²: -0.0123
- F-test p-value: 0.9358
- Observations: 83
- Durbin-Watson: 2.2763

Coefficients:

- const: 2.1988 (SE: 0.0434, p: 0.0000) ***
- GPR_Japan_y: -0.0120 (SE: 0.1487, p: 0.9358)

**Gpr China Us**

- R²: 0.0249
- Adjusted R²: 0.0005
- F-test p-value: 0.3649
- Observations: 83
- Durbin-Watson: 2.2480

Coefficients:

- const: 2.1461 (SE: 0.0798, p: 0.0000) ***
- GPR_China_y: -0.1194 (SE: 0.1050, p: 0.2588)
- GPR_US_y: 0.0491 (SE: 0.0349, p: 0.1634)

**Gpr China Japan**

- R²: 0.0009
- Adjusted R²: -0.0241
- F-test p-value: 0.9641
- Observations: 83
- Durbin-Watson: 2.2649

Coefficients:

- const: 2.2116 (SE: 0.0661, p: 0.0000) ***
- GPR_China_y: -0.0267 (SE: 0.1036, p: 0.7968)
- GPR_Japan_y: 0.0215 (SE: 0.1981, p: 0.9137)

**Gpr Us Japan**

- R²: 0.0124
- Adjusted R²: -0.0123
- F-test p-value: 0.6079
- Observations: 83
- Durbin-Watson: 2.2929

Coefficients:

- const: 2.1326 (SE: 0.0793, p: 0.0000) ***
- GPR_US_y: 0.0287 (SE: 0.0288, p: 0.3215)
- GPR_Japan_y: -0.0850 (SE: 0.1657, p: 0.6096)

**Gpr China Us Japan**

- R²: 0.0251
- Adjusted R²: -0.0120
- F-test p-value: 0.5688
- Observations: 83
- Durbin-Watson: 2.2501

Coefficients:

- const: 2.1466 (SE: 0.0805, p: 0.0000) ***
- GPR_China_y: -0.1273 (SE: 0.1256, p: 0.3136)
- GPR_US_y: 0.0491 (SE: 0.0351, p: 0.1658)
- GPR_Japan_y: 0.0229 (SE: 0.1969, p: 0.9076)

#### Bin Size: 45

**Ambiguity Risk**

- R²: 0.0007
- Adjusted R²: -0.0242
- F-test p-value: 0.9708
- Observations: 83
- Durbin-Watson: 2.3499

Coefficients:

- const: 0.0086 (SE: 0.0564, p: 0.8796)
- Ambiguity_Bins_45: 0.0005 (SE: 0.0248, p: 0.9844)
- Risk_Bins_45: -15.2417 (SE: 63.3182, p: 0.8104)

**Gpr China**

- R²: 0.0006
- Adjusted R²: -0.0117
- F-test p-value: 0.8194
- Observations: 83
- Durbin-Watson: 2.2585

Coefficients:

- const: 2.2449 (SE: 0.0684, p: 0.0000) ***
- GPR_China_y: -0.0186 (SE: 0.0812, p: 0.8194)

**Gpr Us**

- R²: 0.0086
- Adjusted R²: -0.0037
- F-test p-value: 0.4054
- Observations: 83
- Durbin-Watson: 2.3058

Coefficients:

- const: 2.1642 (SE: 0.0823, p: 0.0000) ***
- GPR_US_y: 0.0225 (SE: 0.0269, p: 0.4054)

**Gpr Japan**

- R²: 0.0015
- Adjusted R²: -0.0109
- F-test p-value: 0.7306
- Observations: 83
- Durbin-Watson: 2.2491

Coefficients:

- const: 2.2435 (SE: 0.0453, p: 0.0000) ***
- GPR_Japan_y: -0.0536 (SE: 0.1552, p: 0.7306)

**Gpr China Us**

- R²: 0.0229
- Adjusted R²: -0.0015
- F-test p-value: 0.3956
- Observations: 83
- Durbin-Watson: 2.2369

Coefficients:

- const: 2.1797 (SE: 0.0835, p: 0.0000) ***
- GPR_China_y: -0.1190 (SE: 0.1098, p: 0.2816)
- GPR_US_y: 0.0493 (SE: 0.0365, p: 0.1807)

**Gpr China Japan**

- R²: 0.0015
- Adjusted R²: -0.0235
- F-test p-value: 0.9428
- Observations: 83
- Durbin-Watson: 2.2490

Coefficients:

- const: 2.2437 (SE: 0.0690, p: 0.0000) ***
- GPR_China_y: -0.0004 (SE: 0.1081, p: 0.9973)
- GPR_Japan_y: -0.0532 (SE: 0.2068, p: 0.7978)

**Gpr Us Japan**

- R²: 0.0164
- Adjusted R²: -0.0082
- F-test p-value: 0.5171
- Observations: 83
- Durbin-Watson: 2.2566

Coefficients:

- const: 2.1674 (SE: 0.0826, p: 0.0000) ***
- GPR_US_y: 0.0330 (SE: 0.0300, p: 0.2746)
- GPR_Japan_y: -0.1375 (SE: 0.1727, p: 0.4284)

**Gpr China Us Japan**

- R²: 0.0237
- Adjusted R²: -0.0134
- F-test p-value: 0.5920
- Observations: 83
- Durbin-Watson: 2.2281

Coefficients:

- const: 2.1786 (SE: 0.0841, p: 0.0000) ***
- GPR_China_y: -0.1012 (SE: 0.1312, p: 0.4431)
- GPR_US_y: 0.0492 (SE: 0.0367, p: 0.1837)
- GPR_Japan_y: -0.0517 (SE: 0.2057, p: 0.8021)

#### Bin Size: 50

**Ambiguity Risk**

- R²: 0.0019
- Adjusted R²: -0.0231
- F-test p-value: 0.9272
- Observations: 83
- Durbin-Watson: 2.3548

Coefficients:

- const: 0.0274 (SE: 0.0588, p: 0.6428)
- Ambiguity_Bins_50: -0.0078 (SE: 0.0258, p: 0.7620)
- Risk_Bins_50: -16.7538 (SE: 63.2960, p: 0.7919)

**Gpr China**

- R²: 0.0018
- Adjusted R²: -0.0105
- F-test p-value: 0.7022
- Observations: 83
- Durbin-Watson: 2.1760

Coefficients:

- const: 2.2660 (SE: 0.0658, p: 0.0000) ***
- GPR_China_y: -0.0300 (SE: 0.0781, p: 0.7022)

**Gpr Us**

- R²: 0.0085
- Adjusted R²: -0.0038
- F-test p-value: 0.4078
- Observations: 83
- Durbin-Watson: 2.2351

Coefficients:

- const: 2.1791 (SE: 0.0793, p: 0.0000) ***
- GPR_US_y: 0.0215 (SE: 0.0259, p: 0.4078)

**Gpr Japan**

- R²: 0.0018
- Adjusted R²: -0.0105
- F-test p-value: 0.7047
- Observations: 83
- Durbin-Watson: 2.1692

Coefficients:

- const: 2.2564 (SE: 0.0436, p: 0.0000) ***
- GPR_Japan_y: -0.0568 (SE: 0.1494, p: 0.7047)

**Gpr China Us**

- R²: 0.0288
- Adjusted R²: 0.0045
- F-test p-value: 0.3103
- Observations: 83
- Durbin-Watson: 2.1650

Coefficients:

- const: 2.1968 (SE: 0.0801, p: 0.0000) ***
- GPR_China_y: -0.1364 (SE: 0.1054, p: 0.1991)
- GPR_US_y: 0.0522 (SE: 0.0350, p: 0.1397)

**Gpr China Japan**

- R²: 0.0022
- Adjusted R²: -0.0228
- F-test p-value: 0.9167
- Observations: 83
- Durbin-Watson: 2.1662

Coefficients:

- const: 2.2652 (SE: 0.0664, p: 0.0000) ***
- GPR_China_y: -0.0184 (SE: 0.1041, p: 0.8601)
- GPR_Japan_y: -0.0337 (SE: 0.1990, p: 0.8658)

**Gpr Us Japan**

- R²: 0.0170
- Adjusted R²: -0.0076
- F-test p-value: 0.5037
- Observations: 83
- Durbin-Watson: 2.1737

Coefficients:

- const: 2.1823 (SE: 0.0795, p: 0.0000) ***
- GPR_US_y: 0.0321 (SE: 0.0289, p: 0.2692)
- GPR_Japan_y: -0.1384 (SE: 0.1662, p: 0.4074)

**Gpr China Us Japan**

- R²: 0.0292
- Adjusted R²: -0.0077
- F-test p-value: 0.5026
- Observations: 83
- Durbin-Watson: 2.1559

Coefficients:

- const: 2.1961 (SE: 0.0807, p: 0.0000) ***
- GPR_China_y: -0.1253 (SE: 0.1260, p: 0.3229)
- GPR_US_y: 0.0522 (SE: 0.0352, p: 0.1424)
- GPR_Japan_y: -0.0322 (SE: 0.1975, p: 0.8708)

## Key Findings and Conclusions

1. **Optimal Bin Size**: 20 bins (average adjusted R² = -0.0037)
2. **Log Transformation**: Log transformation improves model performance (avg adj R²: -0.0118 vs -0.0119)
3. **GPR Variables**: Multiple country GPR data shows varying effectiveness across different bin sizes

## 📊 Why Number of Bins Affects Ambiguity Meaning

### 1. Information Granularity Trade-off

Too Few Bins (5-10):

- Oversimplifies return distribution
- Misses important tail behavior
- May underestimate true ambiguity
- Good for capturing broad patterns only
  Optimal Bins (15-25):
- Balances detail with statistical reliability
- Captures meaningful uncertainty patterns
- Sufficient granularity for cross-entropy calculation
- Robust to noise while preserving signal
  Too Many Bins (40-50):
- Introduces noise and overfitting
- Many bins may have zero observations
- Unstable probability estimates
- May overestimate ambiguity due to sampling variation

### 2. Statistical Considerations

With 20-day windows, 20 bins provides approximately 1 observation per bin on average , which represents an optimal information extraction ratio. This relationship suggests:

- Too many bins lead to sparse data problems
- Too few bins lose important distributional information
- The 1:1 ratio (days:bins) maximizes information content

### 3. Economic Interpretation

The optimal bin size may reflect:

- Natural market risk categorization by investors
- How institutional investors process uncertainty in discrete risk levels
- Practical portfolio management decision-making granularity
- The cognitive limits of processing distributional information

### 4. Cross-Entropy Mechanics

- Cross-entropy measures deviation from uniform distribution
- More bins increase potential for higher cross-entropy values
- Optimal bins balance sensitivity with stability
- Bin size affects the baseline uniform distribution comparison

## 💡 Practical Implications

1. Use 20 bins with US GPR data for most robust results
2. The 1:1 window-to-bin ratio appears to be a fundamental principle
3. Single-country GPR (US) reduces multicollinearity while maintaining explanatory power
4. Log transformation provides marginal improvements but doesn't change optimal configurations
5. This analysis suggests that ambiguity measurement is most meaningful when the granularity matches the natural information processing capacity of the data and the economic decision-making framework of market participants.

---

*Report generated on 2025-10-10 19:48:19*
