    libs <- c("tidyverse","Amelia","dplyr","rio","summarytools","readr", "caret", "tidymodels", "skimr", "ggcorrplot", "gridExtra", "RColorBrewer", "tree", "rpart", "rpart.plot", 'rattle', 'cluster', 'factoextra')

    installed_libs <- libs %in% rownames(installed.packages())

    if (any(installed_libs == F)) {
      install.packages(libs[!installed_libs])
    } else {
      print("All libs are already installed")
    }

    ## [1] "All libs are already installed"

    for (lib in libs) {
      library(lib, character.only = TRUE)
    }

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.4     ✔ readr     2.1.5
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ## ✔ ggplot2   3.5.1     ✔ tibble    3.2.1
    ## ✔ lubridate 1.9.3     ✔ tidyr     1.3.1
    ## ✔ purrr     1.0.2     
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors
    ## Loading required package: Rcpp
    ## 
    ## ## 
    ## ## Amelia II: Multiple Imputation
    ## ## (Version 1.8.3, built: 2024-11-07)
    ## ## Copyright (C) 2005-2025 James Honaker, Gary King and Matthew Blackwell
    ## ## Refer to http://gking.harvard.edu/amelia/ for more information
    ## ## 
    ## 
    ## 
    ## Attaching package: 'summarytools'
    ## 
    ## 
    ## The following object is masked from 'package:tibble':
    ## 
    ##     view
    ## 
    ## 
    ## Loading required package: lattice
    ## 
    ## 
    ## Attaching package: 'caret'
    ## 
    ## 
    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift
    ## 
    ## 
    ## ── Attaching packages ────────────────────────────────────── tidymodels 1.2.0 ──
    ## 
    ## ✔ broom        1.0.7     ✔ rsample      1.2.1
    ## ✔ dials        1.3.0     ✔ tune         1.2.1
    ## ✔ infer        1.0.7     ✔ workflows    1.1.4
    ## ✔ modeldata    1.4.0     ✔ workflowsets 1.1.0
    ## ✔ parsnip      1.2.1     ✔ yardstick    1.3.1
    ## ✔ recipes      1.1.0     
    ## 
    ## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
    ## ✖ scales::discard()        masks purrr::discard()
    ## ✖ dplyr::filter()          masks stats::filter()
    ## ✖ recipes::fixed()         masks stringr::fixed()
    ## ✖ dplyr::lag()             masks stats::lag()
    ## ✖ caret::lift()            masks purrr::lift()
    ## ✖ rsample::populate()      masks Rcpp::populate()
    ## ✖ yardstick::precision()   masks caret::precision()
    ## ✖ yardstick::recall()      masks caret::recall()
    ## ✖ yardstick::sensitivity() masks caret::sensitivity()
    ## ✖ yardstick::spec()        masks readr::spec()
    ## ✖ yardstick::specificity() masks caret::specificity()
    ## ✖ recipes::step()          masks stats::step()
    ## ✖ summarytools::view()     masks tibble::view()
    ## • Learn how to get started at https://www.tidymodels.org/start/
    ## 
    ## 
    ## Attaching package: 'gridExtra'
    ## 
    ## 
    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine
    ## 
    ## 
    ## 
    ## Attaching package: 'rpart'
    ## 
    ## 
    ## The following object is masked from 'package:dials':
    ## 
    ##     prune
    ## 
    ## 
    ## Loading required package: bitops
    ## 
    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.5.1 Copyright (c) 2006-2021 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.
    ## 
    ## Welcome! Want to learn more? See two factoextra-related books at https://goo.gl/ve3WBa

# 1. Selection of topic and dataset

A dataset sourced from
[Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)
has been selected as the foundation for the analysis, in accordance with
the project guidelines.

    fetal_df <- read_csv("fetal_health.csv")

    ## Rows: 2126 Columns: 22
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (22): baseline value, accelerations, fetal_movement, uterine_contraction...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

This analysis explores a dataset on fetal cardiotocograms, which
contains 2,126 observations and 22 variables describing various
physiological measurements recorded during pregnancy.

Typ of variables:

-   baseline value -&gt; ratio
-   accelerations -&gt; ratio
-   fetal\_movement -&gt; ratio
-   uterine\_contractions -&gt; ratio
-   light\_decelerations -&gt; ratio
-   severe\_decelerations -&gt; ratio
-   prolongued\_decelerations -&gt; rato
-   abnormal\_short\_term\_variability -&gt; ratio
-   mean\_value\_of\_short\_term\_variability -&gt; ratio
-   percentage\_of\_time\_with\_abnormal\_long\_term\_variability -&gt;
    ratio
-   mean\_value\_of\_long\_term\_variability -&gt; ratio
-   histogram\_width -&gt; ratio
-   histogram\_min -&gt; ratio
-   histogram\_max -&gt; ratio
-   histogram\_number\_of\_peaks -&gt; ratio
-   histogram\_number\_of\_zeroes -&gt; ratio
-   histogram\_mode -&gt; ratio
-   histogram\_mean -&gt; ratio
-   histogram\_median -&gt; ratio
-   histogram\_variance -&gt; ratio
-   histogram\_tendency -&gt; ratio
-   fetal\_health -&gt; ordinal

### *Purpose*

The primary purpose of this analysis is to examine the influence of
various measurements on fetal health. Specifically, the goals are:

-   To explore patterns and relationships among variables using EDA.
-   To identify key predictors of fetal health status.
-   To build predictive models that classify fetal health into one of
    the 3 categories.
-   To test specific hypotheses regarding how different features affect
    fetal condition

### *Research hypothesis*:

1.  There is a relationship between baseline FHR and the number of
    accelerations, fetal movements, and the presence of light
    decelerations.
2.  There is a relationship between abnormal short-term variability and
    the number of severe decelerations, prolonged decelerations, and the
    number of uterine contractions.
3.  There is relationship between fetal health and baseline FHR,
    percentage of time with abnormal long-term variability and the
    number of peaks in the FHR histogram.

# Data Cleaning and EDA

The goal of this stage is to perform descriptive statistics and identify
correlated variables within the dataset.

    glimpse(fetal_df)

    ## Rows: 2,126
    ## Columns: 22
    ## $ `baseline value`                                       <dbl> 120, 132, 133, …
    ## $ accelerations                                          <dbl> 0.000, 0.006, 0…
    ## $ fetal_movement                                         <dbl> 0.000, 0.000, 0…
    ## $ uterine_contractions                                   <dbl> 0.000, 0.006, 0…
    ## $ light_decelerations                                    <dbl> 0.000, 0.003, 0…
    ## $ severe_decelerations                                   <dbl> 0, 0, 0, 0, 0, …
    ## $ prolongued_decelerations                               <dbl> 0.000, 0.000, 0…
    ## $ abnormal_short_term_variability                        <dbl> 73, 17, 16, 16,…
    ## $ mean_value_of_short_term_variability                   <dbl> 0.5, 2.1, 2.1, …
    ## $ percentage_of_time_with_abnormal_long_term_variability <dbl> 43, 0, 0, 0, 0,…
    ## $ mean_value_of_long_term_variability                    <dbl> 2.4, 10.4, 13.4…
    ## $ histogram_width                                        <dbl> 64, 130, 130, 1…
    ## $ histogram_min                                          <dbl> 62, 68, 68, 53,…
    ## $ histogram_max                                          <dbl> 126, 198, 198, …
    ## $ histogram_number_of_peaks                              <dbl> 2, 6, 5, 11, 9,…
    ## $ histogram_number_of_zeroes                             <dbl> 0, 1, 1, 0, 0, …
    ## $ histogram_mode                                         <dbl> 120, 141, 141, …
    ## $ histogram_mean                                         <dbl> 137, 136, 135, …
    ## $ histogram_median                                       <dbl> 121, 140, 138, …
    ## $ histogram_variance                                     <dbl> 73, 12, 13, 13,…
    ## $ histogram_tendency                                     <dbl> 1, 0, 0, 1, 1, …
    ## $ fetal_health                                           <dbl> 2, 1, 1, 1, 1, …

    colnames(fetal_df)[1] <- "baseline_value"

    fetal_df$fetal_health <- as.factor(fetal_df$fetal_health)

Below, the occurrence of NULL values is checked, both for the entire
dataset and individual columns.

    fetal_df %>% 
      summarise(across(everything(), ~any(is.null(.))))

    ## # A tibble: 1 × 22
    ##   baseline_value accelerations fetal_movement uterine_contractions
    ##   <lgl>          <lgl>         <lgl>          <lgl>               
    ## 1 FALSE          FALSE         FALSE          FALSE               
    ## # ℹ 18 more variables: light_decelerations <lgl>, severe_decelerations <lgl>,
    ## #   prolongued_decelerations <lgl>, abnormal_short_term_variability <lgl>,
    ## #   mean_value_of_short_term_variability <lgl>,
    ## #   percentage_of_time_with_abnormal_long_term_variability <lgl>,
    ## #   mean_value_of_long_term_variability <lgl>, histogram_width <lgl>,
    ## #   histogram_min <lgl>, histogram_max <lgl>, histogram_number_of_peaks <lgl>,
    ## #   histogram_number_of_zeroes <lgl>, histogram_mode <lgl>, …

Most of the data are ratio-type variables so to better understand the
data, the following techniques can be used: - mean, - interquartile
range, - mode, - rank tests, - median, - standard deviation, -
confidence intervals for the mean, - t-tests and ANOVA, - Pearson
correlation coefficient, - linear regression.

To analyze ordinal data, such as the variable fetal\_health, the
following methods can be used:

-   median or/and mode to find the central tendancy,
-   percentiles,
-   interquartile deviation,
-   rank tests.

<!-- -->

    fetal_df %>% 
      summary()

    ##  baseline_value  accelerations      fetal_movement     uterine_contractions
    ##  Min.   :106.0   Min.   :0.000000   Min.   :0.000000   Min.   :0.000000    
    ##  1st Qu.:126.0   1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.002000    
    ##  Median :133.0   Median :0.002000   Median :0.000000   Median :0.004000    
    ##  Mean   :133.3   Mean   :0.003178   Mean   :0.009481   Mean   :0.004366    
    ##  3rd Qu.:140.0   3rd Qu.:0.006000   3rd Qu.:0.003000   3rd Qu.:0.007000    
    ##  Max.   :160.0   Max.   :0.019000   Max.   :0.481000   Max.   :0.015000    
    ##  light_decelerations severe_decelerations prolongued_decelerations
    ##  Min.   :0.000000    Min.   :0.000e+00    Min.   :0.0000000       
    ##  1st Qu.:0.000000    1st Qu.:0.000e+00    1st Qu.:0.0000000       
    ##  Median :0.000000    Median :0.000e+00    Median :0.0000000       
    ##  Mean   :0.001889    Mean   :3.293e-06    Mean   :0.0001585       
    ##  3rd Qu.:0.003000    3rd Qu.:0.000e+00    3rd Qu.:0.0000000       
    ##  Max.   :0.015000    Max.   :1.000e-03    Max.   :0.0050000       
    ##  abnormal_short_term_variability mean_value_of_short_term_variability
    ##  Min.   :12.00                   Min.   :0.200                       
    ##  1st Qu.:32.00                   1st Qu.:0.700                       
    ##  Median :49.00                   Median :1.200                       
    ##  Mean   :46.99                   Mean   :1.333                       
    ##  3rd Qu.:61.00                   3rd Qu.:1.700                       
    ##  Max.   :87.00                   Max.   :7.000                       
    ##  percentage_of_time_with_abnormal_long_term_variability
    ##  Min.   : 0.000                                        
    ##  1st Qu.: 0.000                                        
    ##  Median : 0.000                                        
    ##  Mean   : 9.847                                        
    ##  3rd Qu.:11.000                                        
    ##  Max.   :91.000                                        
    ##  mean_value_of_long_term_variability histogram_width  histogram_min   
    ##  Min.   : 0.000                      Min.   :  3.00   Min.   : 50.00  
    ##  1st Qu.: 4.600                      1st Qu.: 37.00   1st Qu.: 67.00  
    ##  Median : 7.400                      Median : 67.50   Median : 93.00  
    ##  Mean   : 8.188                      Mean   : 70.45   Mean   : 93.58  
    ##  3rd Qu.:10.800                      3rd Qu.:100.00   3rd Qu.:120.00  
    ##  Max.   :50.700                      Max.   :180.00   Max.   :159.00  
    ##  histogram_max histogram_number_of_peaks histogram_number_of_zeroes
    ##  Min.   :122   Min.   : 0.000            Min.   : 0.0000           
    ##  1st Qu.:152   1st Qu.: 2.000            1st Qu.: 0.0000           
    ##  Median :162   Median : 3.000            Median : 0.0000           
    ##  Mean   :164   Mean   : 4.068            Mean   : 0.3236           
    ##  3rd Qu.:174   3rd Qu.: 6.000            3rd Qu.: 0.0000           
    ##  Max.   :238   Max.   :18.000            Max.   :10.0000           
    ##  histogram_mode  histogram_mean  histogram_median histogram_variance
    ##  Min.   : 60.0   Min.   : 73.0   Min.   : 77.0    Min.   :  0.00    
    ##  1st Qu.:129.0   1st Qu.:125.0   1st Qu.:129.0    1st Qu.:  2.00    
    ##  Median :139.0   Median :136.0   Median :139.0    Median :  7.00    
    ##  Mean   :137.5   Mean   :134.6   Mean   :138.1    Mean   : 18.81    
    ##  3rd Qu.:148.0   3rd Qu.:145.0   3rd Qu.:148.0    3rd Qu.: 24.00    
    ##  Max.   :187.0   Max.   :182.0   Max.   :186.0    Max.   :269.00    
    ##  histogram_tendency fetal_health
    ##  Min.   :-1.0000    1:1655      
    ##  1st Qu.: 0.0000    2: 295      
    ##  Median : 0.0000    3: 176      
    ##  Mean   : 0.3203                
    ##  3rd Qu.: 1.0000                
    ##  Max.   : 1.0000

    #more detailed summary
    fetal_df %>% 
      select(-fetal_health) %>% 
      skim() %>% 
      mutate(numeric.mean = round(numeric.mean,3),
             numeric.sd = round(numeric.sd,3)) %>%
      select(-n_missing,-complete_rate,-skim_type) %>% 
      View()

------------------------------------------------------------------------

According to the literature (as state in NIH Appendix J “Fetal heart
rate classifications”), the average fetal heart rate ranges between 110
and 160 bpm. The mean value of the variable `baseline_value` in the
dataset falls within this typical normal range.

Other possible states mention in appendix:

1.  Tachycardia baseline FHR &gt; 160 bpm
2.  Bradycardia baseline FHR &lt; 110 bpm

In the dataset, there are no cases of fetal tachycardia, as the maximum
value is 160 bpm, which is the upper limit of the normal range. However,
there are instances of fetal bradycardia. The minimum value is 106 bpm,
which is slightly below the normal threshold.

The majority of the values fall within the range 133.3 ± 9.84 =&gt;
123.5 bpm - 143.1 bpm, which is still in normal range.

IQR = 14 =&gt; middle 50% obs. falls between 126 and 140 bmp.

**Conclusion**

The variable `baseline_value` has a distribution that is approximately
normal, with no significant deviations. Most fetuses have a heart rate
within the normal range. However, there are some cases with values below
110 bpm, which may indicate a need for further diagnostic.

------------------------------------------------------------------------

The `accelerations` represents the number of FHR accelerations per
second (in time). Accelerations are short-term rises in the heart rate
of at least 15 beats per minute, lasting at least 15 seconds. *(for
&gt;32 weeks)*

The median value of accelerations is 0.002, while the Q1 is 0.000,
indicating that at least 25% of the cases showed no accelerations at
all. The values are highly concentrated near 0, with a long tail to the
right (mean &gt; median) what is indicating that the spread occurs only
in one direction. The distribution is not normal – it is skewed. The
maximum value is only 0.019, which confirms that even the highest values
are very small.

IQR = 0.006 =&gt; middle 50% obs. falls between 0 and 0.006.

**Conclusion**

The variable accelerations has a positively skewed distribution, with
the majority of values concentrated near 0. Most fetuses exhibit few or
no heart rate accelerations per second, which may reflect periods of low
activity or fetal sleep. However, a small number of cases show higher
acceleration rates, pulling the mean above the median. These outliers
suggest that while accelerations are generally infrequent, some fetuses
show “liveliness”

------------------------------------------------------------------------

The `fetal_movement` represents the number of fetal movements per
second.

Most of the values are zero—both Q1 and Q2 (the median) are 0. Similar
to the `accelerations` variable, the mean is higher than the median,
accompanied by a relatively large standard deviation. This indicates a
strong right-skewed distribution, with many cases showing no movement,
and a few exhibiting high levels of activity.

IQR = 0.003 =&gt; middle 50% obs. falls between 0 and 0.003.

**Conclusion**

The variable suggests that in most cases, no fetal movements were
recorded, while in a few cases, the movements were significantly
elevated. This may indicate differences in fetal activity levels—a lack
of movement could be physiological but may also be concerning, depending
on the context.

------------------------------------------------------------------------

The `uterine_contractions` represents the number of uterine contractions
per second.

The majority of the values fall within the range 0.004 ± 0.003 =&gt;
0.001 - 0.007.

IQR = 0.005 =&gt; middle 50% obs. falls between 0.002 and 0.007. The
majority of patients showed a certain level of contractions in the data.

In case of this variable, data are more evenly distributed compared to
`fetal_movement`— both Q1 and median are higher, and the distribution is
less skewed.

**Conclusion**

Most cases exhibited a moderate level of contractions. The distribution
is slightly right-skewed, with a small number of cases showing higher
intensity, although these are rare. Values close to 0 are also present
but do not dominate the observations — this indicates that most patients
experienced at least mild contractions.

------------------------------------------------------------------------

The `light_decelerations` represents the number of LDs per second.
Deceleration occurs when the FHR temporarily slows during labor. 3 types
of deceleration may occur: early, late, and variable. While early
decelerations are usually normal, late and variable decelerations can
point to a problem.

More than 50% of the cases had no light decelerations. The values are
highly concentrated near 0, with a long tail to the right (mean &gt;
median) what is indicating that the spread occurs only in one direction.
The distribution is right-skewed. The maximum value is only 0.015, which
confirms that even the highest values are very small.

In case of `severe_decelerations` the vast majority of cases have a
value of 0, indicating no severe decelerations. Only a few isolated
cases recorded values above zero (with a max of 0.001). The histogram is
heavily concentrated at 0, clearly showing that this type of
deceleration is extremely rare.

In case of `prolongued_decelerations` most cases have a value of 0. Only
a few cases reached values above zero (with a max of 0.005). The
distribution is very similar to that of `severe_decelerations`,
indicating that only a vary small number of observations show any
issues.

**Conclusion**

In the analyzed dataset, all types of decelerations occur rarely or very
rarely. Severe and prolonged decelerations appear almost exclusively as
zero values, which may indicate a generally good condition of the
fetuses in the studied group. Light decelerations are slightly more
common, but still not prevalent—they do not clearly indicate risk on
their own, but should be monitored in the context of other indicators,
such as the absence of fetal movement.

------------------------------------------------------------------------

The `abnormal_short_term_variability` represents percentage of time with
abnormal short term variability.

The majority of the values fall within the range 46.99 ± 17.19 =&gt;
29.8% - 64.18%. The minimum recorded value is 12%, and the maximum is
87%. The median is 49 and the mean is 47, which indicates that the data
are symmetrical, with no strong skewness.

**Conclusion**

The distribution shows that in most cases, fetuses spent a significant
proportion of time with abnormal short-term variability. Variability in
fetal heart rate is a key indicator of autonomic nervous system
function. This highlights the importance of closely evaluating this
metric when assessing fetal health status.

------------------------------------------------------------------------

The `histogram_number_of_peaks` represents number of peaks in the exam
histogram.

The mean is 4.068 and the median is 3.000, indicating a slightly
right-skewed distribution. The range is wide, from 0 to 18, which
reflects high variability in the number of peaks between cases. The
standard deviation is 2.949, which is relatively large compared to the
mean, suggesting that the data are quite dispersed. With IQR = 4, it can
be observed that half of the cases fall within the range of 2 to 6
peaks.

**Conclusion**

A low number of peaks may indicate a monotonous fetal heart rate and low
variability, while a very high number of peaks may suggest irregularity
in the heart rate pattern.

------------------------------------------------------------------------

The `fetal_health` represents the classified health status of the fetus.
It is classified into three categories: 1 (Normal), 2 (Suspect), and 3
(Pathological).

    median <- fetal_df %>% 
      summarise(median_health = median(as.numeric(fetal_health))) %>% 
      print()

    ## # A tibble: 1 × 1
    ##   median_health
    ##           <dbl>
    ## 1             1

The median value of 1 indicates that at least half of the observations
in the dataset fall into category 1, representing “Normal” fetal
health.falls into category “Normal” health. This suggests that the
majority of cases in the analyzed group show no significant
abnormalities. The value 1 is also the mode, meaning it is the most
frequently occurring category, confirming that normal fetal condition
dominates the dataset.

    quantile(as.numeric(fetal_df$fetal_health))

    ##   0%  25%  50%  75% 100% 
    ##    1    1    1    1    3

Approximately 75% of obs. fall into the ‘normal’ health category,
indicating that the majority of cases represent typical, healthy fetal
conditions. On the other hand, only around 25% (or less) of observations
are classified as ‘suspacted’ or ‘pathological’, indicating the relative
rarity of such cases in the dataset.

    prop.table(table(fetal_df$fetal_health)) * 100

    ## 
    ##         1         2         3 
    ## 77.845720 13.875823  8.278457

    fetal_df %>% 
      ggplot(aes(x=factor(fetal_health))) +
      geom_bar(stat = "count", width = 0.5, fill = "lightgreen") +
      labs(title = "Occurance in each category", x = "Fetal health", y= "") +
      geom_text(aes(label = after_stat(count)), stat = "count", vjust = -0.2) +
      scale_x_discrete(labels = c("Normal","Suspect","Pathological")) +
      theme_minimal() +
      theme(
        panel.grid.major = element_line(color = "darkgrey", size = 0.25),
        panel.grid.major.x = element_blank(),
      )

    ## Warning: The `size` argument of `element_line()` is deprecated as of ggplot2 3.4.0.
    ## ℹ Please use the `linewidth` argument instead.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    ## generated.

![](FetalHealthProject_files/figure-markdown_strict/unnamed-chunk-9-1.png)

**Conclusion**

Higher values — statuses such as *Suspect* and *Pathological* — appear
less frequently, indicating that these statuses are relatively rare in
this population.

------------------------------------------------------------------------

### *Hypothesis 1 - Regression*

**H₀:** None of the variables (`accelerations`, `fetal_movement`,
`light_decelerations`) have a significant effect on `baseline_value`.
**H₁:** At least one of the variables has a significant effect on
`baseline_value`.

    model_h1 <- fetal_df %>% 
      lm(baseline_value ~ accelerations + fetal_movement + light_decelerations, data = .)

    summary(model_h1)

    ## 
    ## Call:
    ## lm(formula = baseline_value ~ accelerations + fetal_movement + 
    ##     light_decelerations, data = .)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -29.1961  -6.8320   0.1352   7.2936  27.2960 
    ## 
    ## Coefficients:
    ##                      Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)          135.1961     0.3131 431.830  < 2e-16 ***
    ## accelerations       -249.2144    54.6801  -4.558 5.47e-06 ***
    ## fetal_movement        -4.3044     4.5081  -0.955     0.34    
    ## light_decelerations -560.6884    71.4076  -7.852 6.45e-15 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 9.672 on 2122 degrees of freedom
    ## Multiple R-squared:  0.03539,    Adjusted R-squared:  0.03403 
    ## F-statistic: 25.95 on 3 and 2122 DF,  p-value: < 2.2e-16

Since the p-value is less than 2.2e-16, we reject the hypothesis 0. At
least one variable has a significant effect on baseline FHR.

The `accelerations` variable has a statistically significant effect on
`baseline_value` (p &lt; 0.001). An increase in the number of
accelerations is associated with a decrease in baseline fetal heart
rate. The regression coefficient is -249.21, meaning that an increase of
1 unit in the number of accelerations is associated with an average
decrease of 249.21 bpm in the baseline fetal heart rate, assuming all
other variables in the model remain constant.

The `fetal_movement` variable does not have a statistically significant
effect on `baseline_value` (p = 0.34). No relationship was found between
fetal movement and baseline fetal heart rate. This may suggest that
fetal movements are not directly related to FHR in the analyzed dataset.

The `light_decelerations` have a strong and statistically significant
effect on `baseline_value` (p &lt; 0.001). The presence or severity of
decelerations significantly lowers the baseline fetal heart rate. The
regression coefficient of -560.69 indicates a very large effect.

In the regression model, the high t-values for the variables
`accelerations` and `light_decelerations` confirm their significant
effect on `baseline_value`.

R-squared = 0.035 =&gt; The model explains only 3.5% of the variability
in baseline FHR.

This is a low result, meaning the variables have statistical
significance, but low predictive power.

------------------------------------------------------------------------

### *Hypothesis 2 - Regression*

**H₀:** None of the variables (`severe decelerations`,
`prolongued_decelerations`, `uterine_contractions`) have a significant
effect on `abnormal_short_term_variability`. **H₁:** At least one of the
variables has a significant effect on `abnormal_short_term_variability`.

    model_h2 <- lm(abnormal_short_term_variability ~ severe_decelerations + prolongued_decelerations + uterine_contractions, data = fetal_df)

    summary(model_h2)

    ## 
    ## Call:
    ## lm(formula = abnormal_short_term_variability ~ severe_decelerations + 
    ##     prolongued_decelerations + uterine_contractions, data = fetal_df)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -40.724 -13.780   0.443  14.276  38.776 
    ## 
    ## Coefficients:
    ##                            Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                 52.7238     0.6497  81.149  < 2e-16 ***
    ## severe_decelerations     10432.5018  6317.2691   1.651  0.09880 .  
    ## prolongued_decelerations  1868.8879   615.3921   3.037  0.00242 ** 
    ## uterine_contractions     -1388.8554   123.2250 -11.271  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 16.68 on 2122 degrees of freedom
    ## Multiple R-squared:  0.05955,    Adjusted R-squared:  0.05822 
    ## F-statistic: 44.79 on 3 and 2122 DF,  p-value: < 2.2e-16

Since the p-value of the overall model is less than 2.2e-16, we reject
the 0 hypothesis. At least one variable has a statistically significant
effect on abnormal short-term variability.

The `prolongued_decelerations` variable has a statistically significant
effect on `abnormal_short_term_variability` (p = 0.00242). An increase
in the number of prolonged decelerations is associated with an increase
in short-term variability. The regression coefficient is 1868.89,
meaning that a 1 unit increase in prolonged decelerations increases
abnormal short-term variability by an average of 1868.89, assuming all
other variables remain constant.

The `uterine_contractions` variable also has a strong and statistically
significant effect on abnormal\_short\_term\_variability (p &lt; 0.001).
More uterine contractions are associated with a decrease in short-term
variability. The regression coefficient of -1388.86 indicates a
substantial negative relationship.

The `severe_decelerations` variable does not have a statistically
significant effect at the 0.05 level (p = 0.0988). This may suggest a
weak or emerging relationship with short-term variability, although
further analysis would be needed to confirm this.

In the regression model, the high t-values for uterine\_contractions
confirms their strong statistical influence on abnormal short-term
variability.

R-squared = 0.059 → The model explains approximately 5.9% of the
variability in abnormal short-term variability.

This is a relatively low value, indicating that although the variables
are statistically significant, their ability to predict abnormal
short-term variability is limited.

------------------------------------------------------------------------

### *Hypothesis 3 - Kruskal–Wallis test*

1.  For `baseline_value`

**H₀:** The distribution of baseline fetal heart rate is the same across
the groups: healthy, suspicious, and pathological. **H₁:** At least one
group differs in the distribution of baseline fetal heart rate.

    fetal_df %>% 
      kruskal.test(baseline_value ~ fetal_health, data = .)

    ## 
    ##  Kruskal-Wallis rank sum test
    ## 
    ## data:  baseline_value by fetal_health
    ## Kruskal-Wallis chi-squared = 254.33, df = 2, p-value < 2.2e-16

Since the p-value is less than 0.05, we reject the hypothesis 0. This
indicates that there are statistically significant differences in the
distribution of `baseline_value` between at least two of the
`fetal_health` groups.

**Conclusion:** Baseline fetal heart rate differs depending on the fetal
health status.

------------------------------------------------------------------------

1.  For `percentage_of_time_with_abnormal_long_term_variability`

**H₀:** The percentage of time with abnormal long-term variability is
the same across all fetal\_health groups: healthy, suspicious, and
pathological. **H₁:** At least one group differs significantly in terms
of the percentage of time with abnormal long-term variability.

    fetal_df %>% 
      kruskal.test(percentage_of_time_with_abnormal_long_term_variability ~ fetal_health, data = .)

    ## 
    ##  Kruskal-Wallis rank sum test
    ## 
    ## data:  percentage_of_time_with_abnormal_long_term_variability by fetal_health
    ## Kruskal-Wallis chi-squared = 496.07, df = 2, p-value < 2.2e-16

Since the p-value is less than 0.05, we reject the hypothesis 0. This
indicates that there are statistically significant differences in the
percentage of time with abnormal long-term variability between the
fetal\_health groups.

**Conclusion:** Abnormal FHR variability significantly differs between
the fetal health groups.

------------------------------------------------------------------------

1.  For `histogram_number_of_peaks`

**H₀:** The distribution of the number of peaks in the FHR histogram is
the same across the healthy, suspicious, and pathological fetal health
groups. **H₁:** At least one group differs significantly in the
distribution of the number of peaks in the FHR histogram.

    fetal_df %>% 
      kruskal.test(histogram_number_of_peaks ~ fetal_health, data = .)

    ## 
    ##  Kruskal-Wallis rank sum test
    ## 
    ## data:  histogram_number_of_peaks by fetal_health
    ## Kruskal-Wallis chi-squared = 38.77, df = 2, p-value = 3.812e-09

Since the p-value is less than 0.05, we reject the hypothesis 0. There
are statistically significant differences in the number of peaks in the
FHR histogram between the fetal\_health groups.

**Conclusion:** The number of peaks in the FHR histogram significantly
differs between the fetal health states.

------------------------------------------------------------------------

### *Correlation matrix*

    ratio_vars <- fetal_df %>% 
      select(baseline_value, accelerations, fetal_movement, light_decelerations, abnormal_short_term_variability, severe_decelerations, prolongued_decelerations, uterine_contractions, percentage_of_time_with_abnormal_long_term_variability, histogram_number_of_peaks)

    cor_matrix <- cor(ratio_vars)

    ggcorrplot(cor_matrix, hc.order = TRUE, type = "lower",
               lab = TRUE, lab_size = 3, colors = c("red", "white", "blue"),
               title = "Correlation Matrix", ggtheme = theme_minimal())

![](FetalHealthProject_files/figure-markdown_strict/unnamed-chunk-15-1.png)

1.  The correlation between `abnormal_short_term_variability` and
    `percentage_of_time_with_abnormal_long_term_variability` is 0.46,
    indicating a moderate positive correlation. A moderate positive
    correlation exists between short-term variability and the percentage
    of time with abnormal long-term variability. As one increases, the
    other tends to increase.

2.  The correlation between `baseline_value` and
    `abnormal_short_term_variability` is 0.31, indicating a moderate
    positive correlation. As the baseline FHR increases, short-term
    variability appears to increase as well.

3.  The correlation between `acceleration` and
    `percentage_of_time_with_abnormal_long_term_variability` is -0.37,
    indicating a moderate negative correlation. This suggests that as
    the number of accelerations increases, the percentage of time with
    abnormal long-term variability tends to decrease.

4.  The correlation between `light_deceleration` and
    `histogram_number_of_peaks` is 0.4, indicating a moderate positive
    correlation. This suggests that as the number of light decelerations
    increases, the number of peaks in the FHR histogram tends to
    increase as well.

5.  The correlation between `uterin_constraction` and
    `servere_deceleration` is 0.01, indicating a very, very weak
    positive correlation. This suggests that there is almost no
    relationship between the number of uterine contractions and the
    occurrence of severe decelerations. As the number of uterine
    contractions increases, there is no significant change in the number
    of severe decelerations.

*For research hypothesis 1:*

Baseline FHR shows a very weak negative correlation with light
decelerations (-0.16). This means that, as baseline FHR increases, the
occurrence of light decelerations tends to decrease slightly. Baseline
FHR shows a very weak negative correlation with accelerations (-0.08).
This suggests that there is also no strong relationship between baseline
FHR and accelerations. Finally, baseline FHR also shows a very weak
negative correlation with fetal movements (-0.03). This indicates that
there is almost no relationship between baseline FHR and fetal
movements. The relationship is so weak that it can be considered
negligible.

*In conclusion, no statistically significant relationships were found
between the analyzed variables.*

*For research hypothesis 2:*

These coefficients suggest very weak relationships between the variables
analyzed. Particularly, the correlation of -0.23 between abnormal
short-term variability and uterine contractions suggests a weak negative
correlation, meaning that higher short-term variability might be
slightly associated with fewer uterine contractions. However, due to the
low value of this coefficient, the relationship is minimal.

*In conclusion, no statistically significant relationships were found
between the analyzed variables.*

### *Box Plots*

    vars <- c("baseline_value", "accelerations", "fetal_movement", "light_decelerations", "abnormal_short_term_variability", "severe_decelerations", "prolongued_decelerations", "uterine_contractions", "percentage_of_time_with_abnormal_long_term_variability", "histogram_number_of_peaks")

    plot_list <- list()

    custom_colors <- c("1" = "lightgreen", "2" = "#F7A94B", "3" = "#9E3A26")

    for (var in vars) {
      p <- ggplot(fetal_df, aes(x = fetal_health, y = .data[[var]], fill = fetal_health)) +
        geom_boxplot() +
        scale_fill_manual(values = custom_colors)
        labs(y = var, fill = "Fetal Health") +
        theme_minimal() +
        theme(axis.title.x = element_blank(),
              axis.text.x = element_blank())
      
        plot_list[[var]] <- p
    }

    boxplots_10 = grid.arrange(grobs = plot_list, ncol = 5, nrow = 2)

![](FetalHealthProject_files/figure-markdown_strict/unnamed-chunk-16-1.png)

# 3. Decission tree

Based on the observations made above, the dataset contains a lot of
representation for Normal obs. (with few cases of Suspect or
Patological). Therefore, stratification was used to obtain a similar
distribution of variables in both data sets. Lack of stratification may
lead to unbalanced data sets, which may bias the model evaluation.

    set.seed(456)

    fetal_split <- createDataPartition(fetal_df$fetal_health, p=0.75, list = FALSE)

    fetal_training <- fetal_df[fetal_split,]
    fetal_test <- fetal_df[-fetal_split,]

    tree <- fetal_training %>%
      rpart(fetal_health ~., data = ., method = "class")

    tree

    ## n= 1596 
    ## 
    ## node), split, n, loss, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 1596 354 1 (0.778195489 0.139097744 0.082706767)  
    ##    2) mean_value_of_short_term_variability>=0.55 1314 151 1 (0.885083714 0.050228311 0.064687976)  
    ##      4) histogram_mean>=107.5 1231  75 1 (0.939073924 0.051990252 0.008935825) *
    ##      5) histogram_mean< 107.5 83   9 3 (0.084337349 0.024096386 0.891566265) *
    ##    3) mean_value_of_short_term_variability< 0.55 282 126 2 (0.280141844 0.553191489 0.166666667)  
    ##      6) percentage_of_time_with_abnormal_long_term_variability< 68.5 250  94 2 (0.304000000 0.624000000 0.072000000)  
    ##       12) abnormal_short_term_variability< 59.5 57  12 1 (0.789473684 0.210526316 0.000000000)  
    ##         24) histogram_variance< 4.5 50   6 1 (0.880000000 0.120000000 0.000000000) *
    ##         25) histogram_variance>=4.5 7   1 2 (0.142857143 0.857142857 0.000000000) *
    ##       13) abnormal_short_term_variability>=59.5 193  49 2 (0.160621762 0.746113990 0.093264249)  
    ##         26) abnormal_short_term_variability< 79.5 177  33 2 (0.169491525 0.813559322 0.016949153)  
    ##           52) percentage_of_time_with_abnormal_long_term_variability< 6.5 28   9 1 (0.678571429 0.285714286 0.035714286) *
    ##           53) percentage_of_time_with_abnormal_long_term_variability>=6.5 149  13 2 (0.073825503 0.912751678 0.013422819) *
    ##         27) abnormal_short_term_variability>=79.5 16   1 3 (0.062500000 0.000000000 0.937500000) *
    ##      7) percentage_of_time_with_abnormal_long_term_variability>=68.5 32   3 3 (0.093750000 0.000000000 0.906250000) *

    rpart.plot(tree)

    ## Warning: Cannot retrieve the data used to build the model (so cannot determine roundint and is.binary for the variables).
    ## To silence this warning:
    ##     Call rpart.plot with roundint=FALSE,
    ##     or rebuild the rpart model with model=TRUE.

![](FetalHealthProject_files/figure-markdown_strict/unnamed-chunk-19-1.png)

    table(fetal_training$fetal_health, predict(tree, type = "class"))

    ##    
    ##        1    2    3
    ##   1 1219   12   11
    ##   2   78  142    2
    ##   3   12    2  118

The `Normal` status was correctly classified as 1 – 1219 times. It was
misclassified as `Suspected` 12 times and as `Pathological` 11 times.
The `Suspected` status was correctly classified as 2 – 142 times. It was
misclassified as `Normal` 78 times and as `Pathological` 2 times. The
`Pathological` status was correctly classified as 3 – 118 times. It was
misclassified as `Normal` 12 times and as `Suspected` 2 times.

    conf.matrix <- round(prop.table(table(fetal_training$fetal_health, predict(tree, type="class")), 2),3)
    rownames(conf.matrix) <- c("Actually Normal", "Actually Suspected", "Actually Pathological")
    colnames(conf.matrix) <- c("Predicted Normal", "Predicted Suspected", "Predicted Pathological")
    conf.matrix

    ##                        
    ##                         Predicted Normal Predicted Suspected
    ##   Actually Normal                  0.931               0.077
    ##   Actually Suspected               0.060               0.910
    ##   Actually Pathological            0.009               0.013
    ##                        
    ##                         Predicted Pathological
    ##   Actually Normal                        0.084
    ##   Actually Suspected                     0.015
    ##   Actually Pathological                  0.901

While the model finds it most difficult to separate ‘Normal’ from
‘Suspected’, its precision remains high across the board.

    printcp(tree)

    ## 
    ## Classification tree:
    ## rpart(formula = fetal_health ~ ., data = ., method = "class")
    ## 
    ## Variables actually used in tree construction:
    ## [1] abnormal_short_term_variability                       
    ## [2] histogram_mean                                        
    ## [3] histogram_variance                                    
    ## [4] mean_value_of_short_term_variability                  
    ## [5] percentage_of_time_with_abnormal_long_term_variability
    ## 
    ## Root node error: 354/1596 = 0.2218
    ## 
    ## n= 1596 
    ## 
    ##         CP nsplit rel error  xerror     xstd
    ## 1 0.217514      0   1.00000 1.00000 0.046886
    ## 2 0.189266      1   0.78249 0.86158 0.044370
    ## 3 0.087571      2   0.59322 0.59322 0.038148
    ## 4 0.042373      4   0.41808 0.41808 0.032734
    ## 5 0.031073      5   0.37571 0.39266 0.031821
    ## 6 0.014124      6   0.34463 0.34463 0.029985
    ## 7 0.010000      7   0.33051 0.34463 0.029985

Initially (before any splits), the model made 354 misclassifications out
of 1596 cases. This results in an error rate of 22.18% when simply
guessing the most frequent class for all instances (without using any
predictors).

    plotcp(tree)

![](FetalHealthProject_files/figure-markdown_strict/unnamed-chunk-23-1.png)

    tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]

    ## [1] 0.01412429

    bestcp <- tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"]

    tree.pruned <- rpart::prune(tree, cp = bestcp)

    rpart.plot(tree.pruned)

    ## Warning: Cannot retrieve the data used to build the model (so cannot determine roundint and is.binary for the variables).
    ## To silence this warning:
    ##     Call rpart.plot with roundint=FALSE,
    ##     or rebuild the rpart model with model=TRUE.

![](FetalHealthProject_files/figure-markdown_strict/unnamed-chunk-25-1.png)

    table(fetal_training$fetal_health, predict(tree.pruned, type = "class"))

    ##    
    ##        1    2    3
    ##   1 1220   11   11
    ##   2   84  136    2
    ##   3   12    2  118

    conf.matrix_pruned <- round(prop.table(table(fetal_training$fetal_health, predict(tree.pruned, type="class")), 2),3)
    rownames(conf.matrix_pruned) <- c("Actually Normal", "Actually Suspected", "Actually Pathological")
    colnames(conf.matrix_pruned) <- c("Predicted Normal", "Predicted Suspected", "Predicted Pathological")
    conf.matrix_pruned

    ##                        
    ##                         Predicted Normal Predicted Suspected
    ##   Actually Normal                  0.927               0.074
    ##   Actually Suspected               0.064               0.913
    ##   Actually Pathological            0.009               0.013
    ##                        
    ##                         Predicted Pathological
    ##   Actually Normal                        0.084
    ##   Actually Suspected                     0.015
    ##   Actually Pathological                  0.901

As a result of pruning, the precision for the `Normal` class decreased,
whereas the precision for the `Suspected` class improved.

    importance <- as.data.frame(tree.pruned$variable.importance)
    importance$Variable <- rownames(importance)
    colnames(importance)[1] <- "Importance"

    importance <- importance %>% arrange(desc(Importance))

    ggplot(importance, aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_bar(stat = "identity") +
      coord_flip() +
      labs(title = "Predictor Importance",
           x = "var",
           y = "importance") +
      theme_minimal()

![](FetalHealthProject_files/figure-markdown_strict/unnamed-chunk-27-1.png)

    asRules(tree.pruned)

    ## 
    ##  Rule number: 4 [fetal_health=1 cover=1231 (77%) prob=0.94]
    ##    mean_value_of_short_term_variability>=0.55
    ##    histogram_mean>=107.5
    ## 
    ##  Rule number: 12 [fetal_health=1 cover=57 (4%) prob=0.79]
    ##    mean_value_of_short_term_variability< 0.55
    ##    percentage_of_time_with_abnormal_long_term_variability< 68.5
    ##    abnormal_short_term_variability< 59.5
    ## 
    ##  Rule number: 52 [fetal_health=1 cover=28 (2%) prob=0.68]
    ##    mean_value_of_short_term_variability< 0.55
    ##    percentage_of_time_with_abnormal_long_term_variability< 68.5
    ##    abnormal_short_term_variability>=59.5
    ##    abnormal_short_term_variability< 79.5
    ##    percentage_of_time_with_abnormal_long_term_variability< 6.5
    ## 
    ##  Rule number: 7 [fetal_health=3 cover=32 (2%) prob=0.09]
    ##    mean_value_of_short_term_variability< 0.55
    ##    percentage_of_time_with_abnormal_long_term_variability>=68.5
    ## 
    ##  Rule number: 5 [fetal_health=3 cover=83 (5%) prob=0.08]
    ##    mean_value_of_short_term_variability>=0.55
    ##    histogram_mean< 107.5
    ## 
    ##  Rule number: 53 [fetal_health=2 cover=149 (9%) prob=0.07]
    ##    mean_value_of_short_term_variability< 0.55
    ##    percentage_of_time_with_abnormal_long_term_variability< 68.5
    ##    abnormal_short_term_variability>=59.5
    ##    abnormal_short_term_variability< 79.5
    ##    percentage_of_time_with_abnormal_long_term_variability>=6.5
    ## 
    ##  Rule number: 27 [fetal_health=3 cover=16 (1%) prob=0.06]
    ##    mean_value_of_short_term_variability< 0.55
    ##    percentage_of_time_with_abnormal_long_term_variability< 68.5
    ##    abnormal_short_term_variability>=59.5
    ##    abnormal_short_term_variability>=79.5

RULE 1 - class `Normal`:

If:

mean\_value\_of\_short\_term\_variability&gt;=0.55
histogram\_mean&gt;=107.5

then fetal\_health = 1 (Normal) with probability of 94% (1231 cases).

RULE 2 - class `Normal`:

If:

mean\_value\_of\_short\_term\_variability &lt; 0.55
percentage\_of\_time\_with\_abnormal\_long\_term\_variability &lt; 68.5
abnormal\_short\_term\_variability &lt; 59.5

then fetal\_health = 1 (Normal) with probability of 79% (57 cases).

RULE 3 - class `Normal`:

If:

mean\_value\_of\_short\_term\_variability &lt; 0.55
percentage\_of\_time\_with\_abnormal\_long\_term\_variability &lt; 68.5
abnormal\_short\_term\_variability ≥ 59.5
abnormal\_short\_term\_variability &lt; 79.5
percentage\_of\_time\_with\_abnormal\_long\_term\_variability &lt; 6.5

then fetal\_health = 1 (Normal) with probability of 68% (28 cases)

### Training vs Test

    pred_train <- predict(tree.pruned, fetal_training, type = "class")

    train_error <- mean(pred_train != fetal_training$fetal_health)
    print(paste("Error for training set:", round(train_error, 4)))

    ## [1] "Error for training set: 0.0764"

    pred_test <- predict(tree.pruned, fetal_test, type = "class")

    test_error <- mean(pred_test != fetal_test$fetal_health)
    print(paste("Error for test set:", round(test_error, 4)))

    ## [1] "Error for test set: 0.0906"

The difference between the errors is small — only about 1.4 pp. This
indicates that the model generalizes well, showing no signs of
overfitting. It has effectively learned the patterns from the training
set and is able to apply them to the test data.

    conf_matrix <- table(Actual = fetal_test$fetal_health, Predicted = pred_test)
    conf_matrix

    ##       Predicted
    ## Actual   1   2   3
    ##      1 403   4   6
    ##      2  33  40   0
    ##      3   4   1  39

    prop_matrix <- round(prop.table(conf_matrix, 2), 3)
    prop_matrix

    ##       Predicted
    ## Actual     1     2     3
    ##      1 0.916 0.089 0.133
    ##      2 0.075 0.889 0.000
    ##      3 0.009 0.022 0.867

    accuracy <- 1 - test_error
    cat("Error for test set:", round(test_error, 4),
        "\nAccuracy for test set:", round(accuracy, 4)*100, "%")

    ## Error for test set: 0.0906 
    ## Accuracy for test set: 90.94 %

The model achieves high accuracy on the test set — 90.94%, with a
classification error of only 9.06%.

# 4. Clustering

    vars_clustering <- fetal_df %>%
      select(mean_value_of_short_term_variability,
             abnormal_short_term_variability,
             percentage_of_time_with_abnormal_long_term_variability,
             histogram_mean,
             histogram_variance)

    cor_matrix <- cor(vars_clustering)
    cor_matrix

    ##                                                        mean_value_of_short_term_variability
    ## mean_value_of_short_term_variability                                              1.0000000
    ## abnormal_short_term_variability                                                  -0.4307050
    ## percentage_of_time_with_abnormal_long_term_variability                           -0.4702589
    ## histogram_mean                                                                   -0.4454014
    ## histogram_variance                                                                0.5558524
    ##                                                        abnormal_short_term_variability
    ## mean_value_of_short_term_variability                                       -0.43070498
    ## abnormal_short_term_variability                                             1.00000000
    ## percentage_of_time_with_abnormal_long_term_variability                      0.45941272
    ## histogram_mean                                                              0.07455369
    ## histogram_variance                                                         -0.14643382
    ##                                                        percentage_of_time_with_abnormal_long_term_variability
    ## mean_value_of_short_term_variability                                                               -0.4702589
    ## abnormal_short_term_variability                                                                     0.4594127
    ## percentage_of_time_with_abnormal_long_term_variability                                              1.0000000
    ## histogram_mean                                                                                      0.2223206
    ## histogram_variance                                                                                 -0.2815362
    ##                                                        histogram_mean
    ## mean_value_of_short_term_variability                      -0.44540138
    ## abnormal_short_term_variability                            0.07455369
    ## percentage_of_time_with_abnormal_long_term_variability     0.22232062
    ## histogram_mean                                             1.00000000
    ## histogram_variance                                        -0.39942323
    ##                                                        histogram_variance
    ## mean_value_of_short_term_variability                            0.5558524
    ## abnormal_short_term_variability                                -0.1464338
    ## percentage_of_time_with_abnormal_long_term_variability         -0.2815362
    ## histogram_mean                                                 -0.3994232
    ## histogram_variance                                              1.0000000

To perform the clustering analysis, 5 variables were selected:

-   mean\_value\_of\_short\_term\_variability,
-   abnormal\_short\_term\_variability,
-   percentage\_of\_time\_with\_abnormal\_long\_term\_variability,
-   histogram\_mean,
-   histogram\_variance.

These variables were chosen due to their high importance in the decision
tree model and low correlation with one another, making them appropriate
for identifying meaningful clusters in the dataset.

    vars_clustering <- c(
      "mean_value_of_short_term_variability",
      "abnormal_short_term_variability",
      "percentage_of_time_with_abnormal_long_term_variability",
      "histogram_mean",
      "histogram_variance"
    )

    fetal_scaled <- fetal_df %>%
      select(all_of(vars_clustering)) %>%
      mutate(across(everything(), scale))

    folds <- clustering_cv(
      data = fetal_scaled,
      vars = vars_clustering, 
      v = 10,
      cluster_function = "kmeans"
    )

    ## Warning: Using an external vector in selections was deprecated in tidyselect 1.1.0.
    ## ℹ Please use `all_of()` or `any_of()` instead.
    ##   # Was:
    ##   data %>% select(vars_clustering)
    ## 
    ##   # Now:
    ##   data %>% select(all_of(vars_clustering))
    ## 
    ## See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    ## generated.

    k_values <- 2:6

    results <- map_dfr(k_values, function(k) {
      
      folds <- clustering_cv(
        data = fetal_scaled,
        vars = all_of(vars_clustering),
        v = 10,
        cluster_function = "kmeans"
      )
      
      sil_scores <- map_dbl(folds$splits, function(split) {
        train_data <- analysis(split)
        test_data <- assessment(split)
        
        km_model <- kmeans(train_data, centers = k, nstart = 10)
        
        all_data <- rbind(train_data, test_data)
        all_cluster <- c(km_model$cluster,
                         apply(as.matrix(dist(rbind(km_model$centers, test_data)))[-(1:k), 1:k], 1, which.min))
        
        sil <- silhouette(all_cluster, dist(all_data))
        mean(sil[, 3])
      })
      
      tibble(k = k, mean_silhouette = mean(sil_scores))
    })

    mean_sil <- mean(results$mean_silhouette)
    cat("Average silhouette score across 10 folds (2-6 clusters) --->", round(mean_sil, 4))

    ## Average silhouette score across 10 folds (2-6 clusters) ---> 0.2871

Determining optimal number of clusters:

    best_k <- results %>% 
      slice_max(mean_silhouette) %>% pull(k)

    cat("Best number of clusters k =", best_k)

    ## Best number of clusters k = 3

    final_kmeans <- kmeans(fetal_scaled, centers = best_k, nstart = 25)
    fetal_df$cluster_kmeans <- factor(final_kmeans$cluster)

    sil <- silhouette(final_kmeans$cluster, dist(fetal_scaled))
    mean_sil <- mean(sil[, 3])

    cat("Average silhouette score for 3 clusters:", round(mean_sil, 4))

    ## Average silhouette score for 3 clusters: 0.3291

The division into three clusters is moderately good.A value of
approximately 0.33 suggests that most observations were reasonably
assigned to clusters. Some points may lie near cluster boundaries, but
overall, the structure appears meaningful.

    cluster_summary <- fetal_df %>%
      group_by(cluster_kmeans) %>%
      summarise(across(all_of(vars_clustering), mean, .names = "avg_{.col}"))

    cluster_summary

    ## # A tibble: 3 × 6
    ##   cluster_kmeans avg_mean_value_of_short_term_variability avg_abnormal_short_t…¹
    ##   <fct>                                             <dbl>                  <dbl>
    ## 1 1                                                 0.474                   65.2
    ## 2 2                                                 1.25                    43.1
    ## 3 3                                                 2.67                    41.3
    ## # ℹ abbreviated name: ¹​avg_abnormal_short_term_variability
    ## # ℹ 3 more variables:
    ## #   avg_percentage_of_time_with_abnormal_long_term_variability <dbl>,
    ## #   avg_histogram_mean <dbl>, avg_histogram_variance <dbl>

    fetal_df %>% 
      ggplot(aes(x = cluster_kmeans)) +
      geom_bar(width = 0.5, fill = "lightgreen") +
      labs(
        title = "Distribution of observations by cluster",
        x = "Cluster ",
        y = "Number of obs.") +
      geom_text(aes(label = after_stat(count)), stat = "count", vjust = -0.2) +
      theme_minimal() +
      theme(
        panel.grid.major = element_line(color = "darkgrey", size = 0.25),
        panel.grid.major.x = element_blank(),
      )

![](FetalHealthProject_files/figure-markdown_strict/unnamed-chunk-42-1.png)

    library(mclust)

    ## Package 'mclust' version 6.1.1
    ## Type 'citation("mclust")' for citing this R package in publications.

    ## 
    ## Attaching package: 'mclust'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     map

    em_model <- Mclust(fetal_scaled)
    summary(em_model)

    ## ---------------------------------------------------- 
    ## Gaussian finite mixture model fitted by EM algorithm 
    ## ---------------------------------------------------- 
    ## 
    ## Mclust VEV (ellipsoidal, equal shape) model with 9 components: 
    ## 
    ##  log-likelihood    n  df       BIC       ICL
    ##       -5851.451 2126 156 -12898.17 -13252.49
    ## 
    ## Clustering table:
    ##   1   2   3   4   5   6   7   8   9 
    ## 127 297 214 423 261 367  68 288  81

The relatively low BIC and ICL values for the number of clusters
automatically selected by mclust suggest that the model is appropriately
regularized. The BIC value of -14055.71 indicates that the model is not
overfitting, despite having a relatively high number of df.

    fetal_df$cluster_em <- factor(em_model$classification)

    plot(em_model, what = "BIC")

![](FetalHealthProject_files/figure-markdown_strict/unnamed-chunk-44-1.png)
The EM model automatically selected 9 clusters, as the BIC value was
highest for this solution. Higher BIC indicates a better model fit.

    summary(em_model)$modelName

    ## [1] "VEV"

    summary(em_model)$G

    ## [1] 9

    sil_em <- silhouette(as.numeric(fetal_df$cluster_em), dist(fetal_scaled))

    mean_sil_em <- mean(sil_em[, 3])
    cat("Average silhouette score for EM clustering:", round(mean_sil_em, 4))

    ## Average silhouette score for EM clustering: -0.0054

Negative values (such as –0.0054) suggest that many data points are
poorly assigned to clusters — meaning they are closer to a different
cluster than their own. In this configuration, the EM model failed to
identify a meaningful cluster structure. This is a poor result,
indicating that EM clustering does not perform well on this dataset.

    pca <- prcomp(fetal_scaled, scale. = TRUE)
    pca_data <- as.data.frame(pca$x[, 1:2])
    pca_data$cluster <- fetal_df$cluster_em

    pca_data %>% 
      ggplot(aes(x = PC1, y = PC2, color = cluster)) +
      geom_point(alpha = 0.6, size = 2) +
      labs(
        title = "EM Clustering on PCA",
        color = "Cluster"
      ) +
      theme_minimal() + 
      stat_ellipse()

![](FetalHealthProject_files/figure-markdown_strict/unnamed-chunk-47-1.png)
For k-means clustering with 3 clusters, the silhouette score was
0.3291.This is a significantly better result, suggesting that the
k-means method captured the data structure more effectively than the EM
algorithm.

    vars_hypotheses <- c(
      "baseline_value",
      "accelerations",
      "fetal_movement",
      "light_decelerations",
      "abnormal_short_term_variability",
      "severe_decelerations",
      "prolongued_decelerations",
      "uterine_contractions",
      "percentage_of_time_with_abnormal_long_term_variability",
      "histogram_number_of_peaks"
    )

    fetal_df %>%
      group_by(cluster_kmeans) %>%
      summarise(across(all_of(vars_hypotheses),
                       list(mean = mean, sd = sd), .names = "{.col}_{.fn}")) %>% 
      print()

    ## # A tibble: 3 × 21
    ##   cluster_kmeans baseline_value_mean baseline_value_sd accelerations_mean
    ##   <fct>                        <dbl>             <dbl>              <dbl>
    ## 1 1                             140.              8.78           0.000315
    ## 2 2                             133.              9.63           0.00411 
    ## 3 3                             129.              8.09           0.00287 
    ## # ℹ 17 more variables: accelerations_sd <dbl>, fetal_movement_mean <dbl>,
    ## #   fetal_movement_sd <dbl>, light_decelerations_mean <dbl>,
    ## #   light_decelerations_sd <dbl>, abnormal_short_term_variability_mean <dbl>,
    ## #   abnormal_short_term_variability_sd <dbl>, severe_decelerations_mean <dbl>,
    ## #   severe_decelerations_sd <dbl>, prolongued_decelerations_mean <dbl>,
    ## #   prolongued_decelerations_sd <dbl>, uterine_contractions_mean <dbl>,
    ## #   uterine_contractions_sd <dbl>, …

    hyp1_vars <- c("baseline_value", "accelerations", "fetal_movement", "light_decelerations")

    hyp1_summary <- fetal_df %>%
      group_by(cluster_kmeans) %>%
      summarise(across(all_of(hyp1_vars), list(mean = mean, sd = sd), .names = "{.col}_{.fn}")) %>% 
      print()

    ## # A tibble: 3 × 9
    ##   cluster_kmeans baseline_value_mean baseline_value_sd accelerations_mean
    ##   <fct>                        <dbl>             <dbl>              <dbl>
    ## 1 1                             140.              8.78           0.000315
    ## 2 2                             133.              9.63           0.00411 
    ## 3 3                             129.              8.09           0.00287 
    ## # ℹ 5 more variables: accelerations_sd <dbl>, fetal_movement_mean <dbl>,
    ## #   fetal_movement_sd <dbl>, light_decelerations_mean <dbl>,
    ## #   light_decelerations_sd <dbl>

The analysis reveals clear differences between clusters in terms of FHR,
the number of accelerations, and fetal movements. Cluster 2 is
characterized by the highest heart rate but low fetal activity, while
cluster 3 shows the lowest heart rate yet the highest level of
activity.This supports the hypothesis that there is a relationship
between these variables.

    hyp2_vars <- c("abnormal_short_term_variability", "severe_decelerations", 
                   "prolongued_decelerations", "uterine_contractions")

    fetal_df %>%
      group_by(cluster_kmeans) %>%
      summarise(across(all_of(hyp2_vars), list(mean = mean, sd = sd), .names = "{.col}_{.fn}")) %>% 
      print()

    ## # A tibble: 3 × 9
    ##   cluster_kmeans abnormal_short_term_variability_mean abnormal_short_term_vari…¹
    ##   <fct>                                         <dbl>                      <dbl>
    ## 1 1                                              65.2                       11.0
    ## 2 2                                              43.1                       14.5
    ## 3 3                                              41.3                       18.8
    ## # ℹ abbreviated name: ¹​abnormal_short_term_variability_sd
    ## # ℹ 6 more variables: severe_decelerations_mean <dbl>,
    ## #   severe_decelerations_sd <dbl>, prolongued_decelerations_mean <dbl>,
    ## #   prolongued_decelerations_sd <dbl>, uterine_contractions_mean <dbl>,
    ## #   uterine_contractions_sd <dbl>

In cluster 2, the highest mean value of abnormal short-term variability
was observed (65.22), which may indicate a potentially higher risk.
However, both severe decelerations and prolonged decelerations were
close to zero across all clusters, suggesting that these variables did
not differentiate between groups. On the other hand, uterine
contractions were slightly higher in cluster 3.

    hyp3_vars <- c("baseline_value", 
                   "percentage_of_time_with_abnormal_long_term_variability", 
                   "histogram_number_of_peaks")

    fetal_df %>%
      group_by(cluster_kmeans) %>%
      summarise(across(all_of(hyp3_vars), list(mean = mean, sd = sd), .names = "{.col}_{.fn}")) %>% 
      print()

    ## # A tibble: 3 × 7
    ##   cluster_kmeans baseline_value_mean baseline_value_sd percentage_of_time_with…¹
    ##   <fct>                        <dbl>             <dbl>                     <dbl>
    ## 1 1                             140.              8.78                    41.9  
    ## 2 2                             133.              9.63                     2.73 
    ## 3 3                             129.              8.09                     0.563
    ## # ℹ abbreviated name:
    ## #   ¹​percentage_of_time_with_abnormal_long_term_variability_mean
    ## # ℹ 3 more variables:
    ## #   percentage_of_time_with_abnormal_long_term_variability_sd <dbl>,
    ## #   histogram_number_of_peaks_mean <dbl>, histogram_number_of_peaks_sd <dbl>

In general, the characteristics of the clusters align well with the
hypothesized variables, supporting the validity of the proposed
assumptions.

# Data Mining Algorithm

## Random Forest

    library(randomForest)

    ## randomForest 4.7-1.2

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:rattle':
    ## 
    ##     importance

    ## The following object is masked from 'package:gridExtra':
    ## 
    ##     combine

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    train_index <- createDataPartition(fetal_df$fetal_health, p = 0.75, list = FALSE)
    train_data <- fetal_df[train_index, ]
    test_data <- fetal_df[-train_index, ]

    rf_model <- randomForest(fetal_health ~ ., data = train_data, ntree = 500, importance = TRUE)

    pred <- predict(rf_model, newdata = test_data)

    confusionMatrix(pred, test_data$fetal_health)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   1   2   3
    ##          1 404  17   2
    ##          2   8  52   2
    ##          3   1   4  40
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9358          
    ##                  95% CI : (0.9115, 0.9552)
    ##     No Information Rate : 0.7792          
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.8192          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.2367          
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3
    ## Sensitivity            0.9782  0.71233  0.90909
    ## Specificity            0.8376  0.97812  0.98971
    ## Pos Pred Value         0.9551  0.83871  0.88889
    ## Neg Pred Value         0.9159  0.95513  0.99175
    ## Prevalence             0.7792  0.13774  0.08302
    ## Detection Rate         0.7623  0.09811  0.07547
    ## Detection Prevalence   0.7981  0.11698  0.08491
    ## Balanced Accuracy      0.9079  0.84522  0.94940

The classification accuracy reached 95.06%, indicating a very high
effectiveness of the model in predicting fetal health status. A Kappa
coefficient of 0.86 confirms strong agreement between the model’s
predictions and the actual labels, meaning the model performs
significantly better than random classification.

For `fetal_health` 1 (Normal), the model achieved very high sensitivity
and precision, indicating that it makes almost no errors in identifying
healthy cases. For `fetal_health` 2 (Suspected), sensitivity was lower
(0.8644), but precision was very high (0.8500), suggesting that the
model is cautious yet effective in detecting suspected cases. For
`fetal_health` 3 (Pathological), the model achieved good sensitivity
(0.85714) and very high precision (0.90909), which is especially
important in the context of identifying high-risk cases.

    varImpPlot(rf_model)

![](FetalHealthProject_files/figure-markdown_strict/unnamed-chunk-53-1.png)

    fetal_df_2 <- fetal_df %>% 
      select(-cluster_kmeans)

    train_index_2 <- createDataPartition(fetal_df_2$fetal_health, p = 0.8, list = FALSE)
    train_data_2 <- fetal_df[train_index_2, ]
    test_data_2 <- fetal_df[-train_index_2, ]

    rf_model_2 <- randomForest(fetal_health ~ ., data = train_data_2, ntree = 500, importance = TRUE)

    pred <- predict(rf_model_2, newdata = test_data)

    confusionMatrix(pred, test_data$fetal_health)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   1   2   3
    ##          1 411   3   0
    ##          2   1  70   1
    ##          3   1   0  43
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9887          
    ##                  95% CI : (0.9755, 0.9958)
    ##     No Information Rate : 0.7792          
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.969           
    ##                                           
    ##  Mcnemar's Test P-Value : 0.3916          
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3
    ## Sensitivity            0.9952   0.9589  0.97727
    ## Specificity            0.9744   0.9956  0.99794
    ## Pos Pred Value         0.9928   0.9722  0.97727
    ## Neg Pred Value         0.9828   0.9934  0.99794
    ## Prevalence             0.7792   0.1377  0.08302
    ## Detection Rate         0.7755   0.1321  0.08113
    ## Detection Prevalence   0.7811   0.1358  0.08302
    ## Balanced Accuracy      0.9848   0.9773  0.98761

The classification accuracy reached 93.88%, indicating a very high
effectiveness of the model in predicting fetal health status. A Kappa
coefficient of 0.83 confirms strong agreement between the model’s
predictions and the actual labels, meaning the model performs
significantly better than random classification.

For `fetal_health` 1 (Normal), the model achieved very high sensitivity
and precision, indicating that it makes almost no errors in identifying
healthy cases. For `fetal_health` 2 (Suspected), sensitivity was lower
(0.7458), but precision was very high (0.8800), suggesting that the
model is cautious yet effective in detecting suspected cases. For
`fetal_health` 3 (Pathological), the model achieved good sensitivity
(0.91429) and high precision (0.84211), which is especially important in
the context of identifying high-risk cases.
