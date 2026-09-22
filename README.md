# Instructor Effectiveness Analysis and Modeling

A data science project that measures and predicts instructor effectiveness using online learning batch data.

Instructors are classified into three relative tiers: **Low**, **Medium**, and **High**.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Objectives](#objectives)
3. [Dataset](#dataset)
4. [Project Workflow](#project-workflow)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Effectiveness Score](#effectiveness-score)
7. [Robustness Checks](#robustness-checks)
8. [Machine Learning](#machine-learning)
9. [Key Findings](#key-findings)
10. [Limitations](#limitations)
11. [Project Structure](#project-structure)
12. [How to Run](#how-to-run)
13. [License & Author](#license--author)

---

## Problem Statement

Online learning platforms collect many batch-level signals such as completion rates, quiz scores, engagement activity, and learner feedback. Individually, these metrics are useful-but together they are hard to turn into a clear, fair view of **instructor performance**.

Two practical challenges appear:

- **No single label** for teaching quality exists in the data
- **Course differences** can make some instructors look stronger or weaker for reasons unrelated to teaching

This project builds a transparent effectiveness measure from available metrics, then uses machine learning to predict effectiveness tiers from leading indicators.

---

## Objectives

1. Understand the structure and quality of the batch-level dataset
2. Identify which metrics relate most strongly at the instructor level
3. Define a clear, weighted **effectiveness score** across outcomes, engagement, and feedback
4. Adjust for course differences so instructors are compared more fairly
5. Assign instructors to **Low / Medium / High** tiers
6. Train and compare classical ML models to predict those tiers
7. Document limitations so results are interpreted responsibly

**Important:** The effectiveness score is an **operational definition** created for this analysis. It is not ground-truth teaching quality and should not be used alone for hiring, promotion, compensation, or disciplinary decisions.

---

## Dataset

| Detail | Value |
|--------|-------|
| File | `Data/instructor_effectiveness_dataset_2000_rows - instructor_effectiveness_dataset_2000_rows.csv.csv` |
| Records | 2,000 batches |
| Instructors | 120 |
| Courses | 25 |
| Columns | 12 |
| Missing values | None |
| Duplicate batch IDs | None |

Each row represents one **batch** taught by one instructor in one course. Multiple batches can belong to the same instructor.

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| `batch_id` | ID | Unique batch identifier |
| `instructor_id` | ID | Instructor identifier |
| `course_id` | ID | Course identifier |
| `completion_rate` | Outcome | Share of learners who completed the batch |
| `avg_score_improvement` | Outcome | Average improvement in learner scores |
| `avg_quiz_score` | Outcome | Average quiz performance |
| `dropout_rate` | Diagnostic | Share of learners who dropped out |
| `avg_watch_time` | Engagement | Average content watch time |
| `assignment_submission_rate` | Engagement | Share of assignments submitted |
| `forum_activity_rate` | Engagement | Forum participation rate |
| `avg_feedback_score` | Feedback | Average learner feedback score |
| `feedback_response_rate` | Feedback | Feedback / survey response rate |

### Basic ranges (approximate)

| Metric | Typical range in data |
|--------|------------------------|
| Completion rate | 0.30 – 0.98 |
| Dropout rate | 0.02 – 0.70 |
| Quiz score | ~40 – 100 |
| Feedback score | ~2.6 – 5.0 |
| Watch / submission / response rates | 0 – 1 |

Some metrics appear **capped** at boundaries (for example, completion never below 0.30, dropout never above 0.70). This is noted during analysis and treated carefully.

---

## Project Workflow

```
Raw batch data
      │
      ▼
Data understanding & quality checks
      │
      ▼
Exploratory analysis (batch vs instructor level)
      │
      ▼
Course-normalized domain scores
      │
      ▼
Weighted effectiveness score
      │
      ▼
Aggregate to instructor level → Low / Medium / High tiers
      │
      ▼
Robustness checks + ML prediction
```

---

## Exploratory Data Analysis

### Data quality

- No missing values
- No duplicate batch IDs
- 120 unique instructors and 25 unique courses
- Instructors teach different numbers of batches; averages are used to represent typical performance

### Batch level vs instructor level

At the **batch level**, relationships between metrics are often moderate or weak.

After aggregating to the **instructor level** (averaging metrics per instructor), relationships become much stronger. This suggests that differences between instructors are more stable than noise in individual batches.

### Important correlation finding

`completion_rate` and `dropout_rate` are almost mirror measures (very strong negative correlation, about −0.95).

Using both in the main score would count nearly the same information twice. Therefore:

- **Completion** is included in the primary effectiveness score
- **Dropout** is kept only as a diagnostic / validation check

### Course context

Instructor-level variation is larger than course-level variation for most metrics, which supports focusing on instructors. Course differences are still present, so metrics are normalized **within each course** before scoring.

### Latent structure (PCA)

Principal Component Analysis at the instructor level shows a strong common dimension across metrics. A simple standardized composite score also aligns closely with the first principal component. This supports the idea that many metrics move together as part of a shared instructor-level pattern—but it does not prove a true “teaching quality” variable exists.

---

## Effectiveness Score

### Domain groups and weights

| Domain | Weight | Why it matters | Metrics |
|--------|--------|----------------|---------|
| Outcomes | 50% | Student results are the strongest signal of impact | Completion rate, score improvement, quiz score |
| Engagement | 30% | Shows whether learners stay active and participate | Watch time, assignment submission, forum activity |
| Feedback | 20% | Captures learner perception and responsiveness | Feedback score, feedback response rate |

Outcomes receive the highest weight because they reflect learning results most directly. Engagement and feedback provide supporting evidence.

### Scoring steps

1. **Normalize within course**  
   For each metric, compare a batch to other batches in the same course (mean / standard deviation). This reduces the risk of treating hard or easy courses as instructor differences.

2. **Build domain scores**  
   Average the normalized metrics inside each domain (outcomes, engagement, feedback).

3. **Compute batch effectiveness**  
   ```
   Effectiveness = (0.5 × Outcome score)
                 + (0.3 × Engagement score)
                 + (0.2 × Feedback score)
   ```

4. **Aggregate to instructors**  
   Average batch effectiveness (and domain scores) for each instructor.

5. **Create tiers**  
   Split instructors into three equal groups using tertiles:
   - **Low**
   - **Medium**
   - **High**

These tiers are **relative** within this dataset, not absolute industry standards.

### Why this definition

| Design choice | Reason |
|---------------|--------|
| Three domains | Covers results, behavior, and perception |
| Higher weight on outcomes | Prioritizes learner performance |
| Exclude dropout from main score | Avoids double-counting with completion |
| Course normalization | Fairer comparison across courses |
| Instructor aggregation | Matches the unit we care about evaluating |
| Relative tiers | Useful when no ground-truth labels exist |

---

## Robustness Checks

To test whether the ranking depends too heavily on one modeling choice, the project compared:

| Check | What was compared |
|-------|-------------------|
| Raw vs course-normalized | Scores without vs with within-course adjustment |
| Batch-weighted vs course-balanced | Simple instructor mean vs mean of course-level means |
| Composite vs PCA | Weighted score vs first principal component |

**Result:** Instructor rankings and tier structure remained broadly stable. This increases confidence that the score is a reasonable operational measure for this dataset—not that it is a perfect measure of teaching quality.

---

## Machine Learning

### Prediction setup

| Item | Detail |
|------|--------|
| Unit of prediction | Instructor |
| Sample size | 120 instructors |
| Target | Low / Medium / High (from the effectiveness score) |
| Train / test split | 80% / 20%, stratified |
| Primary metric | Macro-F1 (equal weight across all three classes) |

### Two model setups

| Model type | Features used | Purpose |
|------------|---------------|---------|
| **Main model** | Engagement + feedback only | Predict tiers from leading indicators without using outcome variables that built the target |
| **Reference model** | Outcomes + engagement + feedback | Upper-bound / leakage check; expected to perform well because it sees score ingredients |

Using outcomes in the main model would mostly reconstruct the formula used to create the target. That would look accurate but would not show real predictive value. The main model therefore focuses on **engagement and feedback**.

### Algorithms compared

- Logistic Regression
- Random Forest
- Gradient Boosting

Models are compared using Macro-F1, precision, recall, and confusion matrices.

### Interpretation notes

- Feature importance can highlight which inputs the model relies on
- High importance for variables that also define the score may simply show reconstruction, not causal teaching drivers
- With only 120 instructors, performance estimates can be unstable and should be treated cautiously

---

## Key Findings

1. **Instructor-level patterns are stronger than batch-level patterns.** Averaging across batches reveals more consistent differences between instructors.

2. **Completion and dropout are redundant.** They behave like opposite sides of the same outcome, so only completion enters the primary score.

3. **Course adjustment matters.** Normalizing within course helps separate course difficulty from instructor performance.

4. **A shared latent structure exists.** PCA and the composite score move together, supporting a single underlying instructor-level factor in this data.

5. **Prediction is possible but limited.** Engagement and feedback can help estimate tiers, but the small instructor sample and constructed target limit how far results should be generalized.

6. **Responsible use is required.** This work is for analytical insight, not automated personnel decisions.

---

## Limitations

| Limitation | Impact |
|------------|--------|
| Constructed target | Models predict a defined score, not independent teaching quality |
| Small instructor sample (n = 120) | Model metrics can vary and may not generalize |
| Capped / bounded metrics | Extreme values may be truncated or compressed |
| Feedback bias | Response patterns may not represent all learners |
| Confounding | Course policy, difficulty, and learner mix can still influence metrics |
| No time dimension | Changes in instructor performance over time are not modeled |
| Metric gaming risk | In real use, people may optimize reported metrics rather than learning quality |

### What would improve future work

- Pre-course learner ability / prior performance
- Learner demographics and course difficulty labels
- Instructor experience and qualifications
- Qualitative feedback text
- Time-based or longitudinal batch data
- Independent quality labels (for example, expert review)

---

## Project Structure

```
Instructor-Effectiveness-Analysis-and-Modeling/
├── Data/
│   └── instructor_effectiveness_dataset_2000_rows - ...csv.csv
├── Notebook/
│   └── Assignment - DS.ipynb          # Full analysis and modeling workflow
├── LICENSE
└── README.md
```

The notebook covers the full pipeline:

1. Data understanding  
2. Exploratory data analysis  
3. Effectiveness score definition  
4. Robustness checks  
5. Machine learning models  
6. Interpretation and limitations  

---

## How to Run

### Requirements

- Python 3.9 or later
- pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Open the notebook

```bash
jupyter notebook "Notebook/Assignment - DS.ipynb"
```

### Data path

If the notebook still points to a Kaggle path, update it to the local file:

```python
df = pd.read_csv(
    "Data/instructor_effectiveness_dataset_2000_rows - instructor_effectiveness_dataset_2000_rows.csv.csv"
)
```

---

## License & Author

See [LICENSE](LICENSE) for license details.

**Author:** Atif Mazhar
