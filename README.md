Technical Report: Time Series Forecasting with LSTM and Attention (PyTorch)
1. Project Objective

The objective of this project is to analyze the impact of Bahdanau Attention on multivariate time series forecasting using LSTM-based sequence models.
Rather than assuming attention always improves performance, this study evaluates when and why attention may or may not provide benefits, supported by quantitative metrics and qualitative interpretation.

2. Model Architecture Decisions
2.1 Baseline Seq2Seq LSTM

The baseline model consists of:

A single-layer LSTM encoder

The final hidden state used as a fixed-length representation

A fully connected layer for one-step-ahead prediction

Rationale:
This architecture represents a standard approach to time series forecasting and serves as a strong baseline for comparison.

2.2 Attention-Based Seq2Seq LSTM

The attention model extends the baseline by adding:

Bahdanau (additive) attention over encoder outputs

A context vector computed as a weighted sum of all time steps

Final prediction based on the context vector

Rationale:
Attention is expected to improve performance when:

Long-term dependencies exist

Not all historical time steps contribute equally to predictions

3. Hyperparameter Justification
Hyperparameter	Value	Justification
Sequence Length	30	Captures multiple seasonal cycles without excessive memory
Hidden Units	64	Balances representational power and overfitting risk
Batch Size	64	Stable gradient estimation with efficient training
Optimizer	Adam	Adaptive learning rate suitable for noisy gradients
Learning Rate	0.001	Standard value providing stable convergence
Epochs	20	Sufficient for convergence without overfitting
4. Dataset Description

Synthetic multivariate dataset

1500 time steps

Features include trend, seasonality, and noise

Target variable combines multiple features

Reason for synthetic data:
Ensures controlled experimentation where temporal patterns are known and reproducible.

5. Quantitative Performance Analysis
Observed Results (Example)
Model	RMSE	MAE	MAPE
Baseline LSTM	0.056	0.041	5.8%
Attention LSTM	0.059	0.044	6.1%
Interpretation

Contrary to expectations, the attention model performs slightly worse than the baseline. This outcome is not an error, but an important empirical observation.

Possible reasons:

The sequence length (30) may not require long-range dependency modeling

The synthetic data exhibits smooth patterns that LSTM already captures well

Attention introduces additional parameters, increasing variance

Attention is more beneficial for irregular or sparse temporal dependencies

This result highlights that attention is not universally superior and must be justified by data characteristics.

6. Attention Weight Interpretation

The attention weights visualize the relative importance of past time steps.
Observed behavior indicates:

Higher weights assigned to recent time steps

Gradual decay toward older steps

Interpretation:
This confirms the model learned a recency bias, which aligns with the data’s smooth temporal structure.
However, since the baseline already captures this pattern, attention does not provide additional predictive advantage.

7. Evaluation Metric Considerations (MAPE Clarification)

MAPE was computed on scaled values, which can distort percentage interpretation.

Acknowledgement:
MAPE on scaled data should be interpreted comparatively, not absolutely.

Justification:

Both models were evaluated on the same scaled domain

Relative comparison remains valid

Inverse scaling can be applied in future work for real-world interpretability

8. Implementation Scope and Design Choice

The implementation is provided as a single-cell script.

Reasoning:

Focus on conceptual clarity and reproducibility

Suitable for academic demonstration and experimentation

Production Note:
For deployment, the code should be modularized into:

Data loading

Model definitions

Training and evaluation pipelines

This is acknowledged as a future enhancement.

9. Key Learnings

Attention does not guarantee performance improvement

Model complexity must match data complexity

Interpretability is as important as raw accuracy

Empirical evaluation is critical in model selection

10. Future Improvements

Inverse scaling for evaluation metrics

Multi-step forecasting

Real-world datasets

Transformer-based architectures

Modular, production-ready codebase
