# Instructor Effectiveness Analysis and Modeling

Predict instructor effectiveness (**Low / Medium / High**) from batch-level learning metrics using exploratory analysis, a transparent composite score, robustness checks, and classical ML models.

---

## Overview

Online learning platforms generate rich batch-level signals—completion, quiz performance, engagement, and feedback—but these are rarely turned into a clear, instructor-level effectiveness measure.

This project:

1. Explores **2,000 batch records** across **120 instructors** and **25 courses**
2. Builds an **operational effectiveness score** from outcomes, engagement, and feedback
3. Aggregates scores to the **instructor level** and assigns relative tiers
4. Trains ML models to predict tiers from **leading indicators** (engagement + feedback), while avoiding direct leakage from outcome variables used in the target

> The effectiveness score is an **operational definition** for analysis—not ground-truth teaching quality. It should not be used alone for hiring, promotion, or disciplinary decisions.

---

## Dataset

| Attribute | Value |
|-----------|-------|
| File | `Data/instructor_effectiveness_dataset_2000_rows - instructor_effectiveness_dataset_2000_rows.csv.csv` |
| Rows | 2,000 batches |
| Instructors | 120 |
| Courses | 25 |
| Missing values | None |
| Duplicate batch IDs | None |

### Features

| Column | Description |
|--------|-------------|
| `batch_id` | Unique batch identifier |
| `instructor_id` | Instructor identifier |
| `course_id` | Course identifier |
| `completion_rate` | Share of learners who completed the batch |
| `avg_score_improvement` | Average score improvement |
| `avg_quiz_score` | Average quiz performance |
| `dropout_rate` | Share of learners who dropped out |
| `avg_watch_time` | Average content watch time (normalized) |
| `assignment_submission_rate` | Assignment submission rate |
| `forum_activity_rate` | Forum participation rate |
| `avg_feedback_score` | Average learner feedback score |
| `feedback_response_rate` | Feedback / survey response rate |

---

## Methodology

### 1. Data understanding & EDA

- Inspect distributions, capping/boundaries, and correlations
- Compare **batch-level** vs **instructor-level** relationships
- Key finding: correlations are much stronger after aggregating to instructors
- `completion_rate` and `dropout_rate` are nearly mirror measures (~−0.95), so only **completion** enters the primary score

### 2. Effectiveness score

Metrics are grouped into three domains and **normalized within course** before scoring:

| Domain | Weight | Metrics |
|--------|--------|---------|
| Outcomes | 50% | `completion_rate`, `avg_score_improvement`, `avg_quiz_score` |
| Engagement | 30% | `avg_watch_time`, `assignment_submission_rate`, `forum_activity_rate` |
| Feedback | 20% | `avg_feedback_score`, `feedback_response_rate` |

\[
\text{effectiveness} = 0.5 \cdot \text{outcome} + 0.3 \cdot \text{engagement} + 0.2 \cdot \text{feedback}
\]

Batch scores are then averaged to the instructor level and split into **Low / Medium / High** tiers via tertiles (`pd.qcut`).

### 3. Robustness checks

Stability is checked against:

- Raw (non course-normalized) aggregation
- Course-balanced aggregation
- PCA latent structure at instructor level

Rankings remain broadly consistent across these alternatives.

### 4. Machine learning

| Item | Detail |
|------|--------|
| Unit of prediction | Instructor (n = 120) |
| Target | Low / Medium / High |
| Main model features | Engagement + feedback only (leading indicators) |
| Reference model | Outcomes + engagement + feedback (expected to reconstruct the score) |
| Models compared | Logistic Regression, Random Forest, Gradient Boosting |
| Primary metric | **Macro-F1** (equal weight across three classes) |

The reference model is used only as a leakage / upper-bound check. The main model excludes outcome variables used to construct the target.

---

## Project structure

```
Instructor-Effectiveness-Analysis-and-Modeling/
├── Data/
│   └── instructor_effectiveness_dataset_2000_rows - ...csv.csv
├── Notebook/
│   └── Assignment - DS.ipynb
├── LICENSE
└── README.md
```

---

## Getting started

### Requirements

- Python 3.9+
- pandas, numpy, matplotlib, seaborn, scikit-learn

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the notebook

```bash
jupyter notebook "Notebook/Assignment - DS.ipynb"
```

Update the data path in the notebook if needed:

```python
df = pd.read_csv("Data/instructor_effectiveness_dataset_2000_rows - instructor_effectiveness_dataset_2000_rows.csv.csv")
```

---

## Key findings

- Instructor-level differences are more persistent than batch-level noise
- Completion and dropout are redundant; using both would double-count the same signal
- Course normalization reduces confounding course difficulty with instructor performance
- A common latent dimension (PCA) aligns strongly with the composite score
- With only 120 instructors, ML results should be interpreted cautiously

---

## Limitations

- Constructed target, not independent teaching-quality labels
- Small instructor sample (n = 120)
- Several metrics appear capped at boundaries
- Feedback may reflect selection or response bias
- Course context can still influence observed performance
- Not suitable as a standalone HR / evaluation tool

---

## License

See [LICENSE](LICENSE) for details.

---

**Author:** Atif Mazhar
