# Justification of MLP Architecture for Privacy-Accuracy Trade-off Analysis

## Executive Summary

This document justifies the decision to use a **Multi-Layer Perceptron (MLP)** as the baseline architecture for privacy-preserving technique evaluation, despite LSTM being identified as the best-performing model for these tasks. The choice prioritizes **precise evaluation of privacy-accuracy trade-offs** over maximum baseline accuracy, enabling systematic analysis of Differential Privacy (DP) and Federated Learning (FL) impacts.

## 1. Context and Motivation

### 1.1 Initial Architecture Exploration

During the initial development phase, we evaluated multiple architectures:

- **CNN+LSTM**: Complex architecture with ~100K+ parameters
- **LSTM-only**: Simplified temporal model with ~50-211K parameters
- **MLP**: Feedforward network with features-only input

**Performance Results:**

| Dataset | Architecture | Accuracy | Training Time | Parameters |
|---------|-------------|----------|---------------|------------|
| Sleep-EDF | CNN+LSTM | 85.39% | ~XX min | ~311K |
| Sleep-EDF | LSTM-only | 85.53% | ~XX min | ~50K |
| WESAD | CNN+LSTM | 72.57% | 3.7s | ~100K+ |
| WESAD | LSTM-only | 82.70% | ~10s | 211K |

**Key Finding:** LSTM-only achieved superior or comparable accuracy compared to CNN+LSTM, making it the optimal choice for pure performance.

### 1.2 Problem with LSTM for Privacy Analysis

However, when evaluating privacy-preserving techniques, we identified critical limitations:

**Issue 1: Training Time Complexity**
- LSTM training is computationally expensive due to sequential processing
- Multiple experimental runs (different seeds, privacy budgets, client configurations) become prohibitively slow
- Limited ability to conduct comprehensive hyperparameter sweeps

**Issue 2: Temporal Pattern Interference**
- LSTM's temporal dependencies create complex gradient flows
- Privacy mechanisms (DP noise, FL aggregation) interact non-linearly with recurrent dynamics
- Difficult to isolate whether accuracy degradation comes from:
  - Privacy mechanisms themselves
  - Interaction between privacy and temporal modeling
  - Numerical instability in recurrent computations

**Issue 3: Evaluation Precision**
- Temporal patterns in LSTM make it harder to precisely measure privacy impact
- Baseline accuracy variations across runs mask subtle privacy effects
- Need for cleaner signal-to-noise ratio in privacy-accuracy trade-off analysis

## 2. MLP Architecture Rationale

### 2.1 Design Principles

The MLP architecture was designed with the following principles:

1. **Features-Only Input**: Accepts pre-extracted features (no temporal modeling)
2. **Fast Training**: Feedforward only, enabling rapid experimentation
3. **DP-Compatible**: Uses LayerNorm (not BatchNorm) for Differential Privacy compatibility
4. **Unified Across Datasets**: Same architecture structure for both Sleep-EDF and WESAD

### 2.2 Architecture Details

**Unified MLP Structure:**
```
Input (features) → Linear → LayerNorm → ReLU → Dropout
                  → Linear → LayerNorm → ReLU → Dropout
                  → Linear (output)
```

**Key Components:**
- **LayerNorm**: Enables per-sample gradient computation required for DP-SGD
- **No BatchNorm**: Incompatible with DP (requires batch statistics)
- **Flexible Hidden Dimensions**: Configurable per dataset (typically [128, 64])
- **Average Pooling**: Handles temporal inputs by averaging over time dimension

### 2.3 Computational Advantages

**Training Efficiency:**
- **No sequential processing**: All operations are parallelizable
- **Faster convergence**: Simpler optimization landscape
- **Lower memory footprint**: No hidden state storage
- **Rapid experimentation**: Enables extensive hyperparameter sweeps

**Inference Speed:**
- **Single forward pass**: No unrolling over time steps
- **Lower latency**: Critical for mobile/edge deployment scenarios
- **Reduced computational cost**: Important for resource-constrained devices

## 3. Privacy-Accuracy Trade-off Analysis Benefits

### 3.1 Precise Impact Measurement

**With MLP:**
- Clean separation between privacy mechanisms and model complexity
- Direct measurement of privacy impact on accuracy
- Reduced confounding factors from temporal modeling
- More interpretable degradation patterns

**With LSTM:**
- Temporal dynamics interact with privacy noise
- Harder to distinguish privacy effects from recurrent instability
- Gradient flow complexity obscures privacy impact
- Baseline variance masks subtle privacy effects

### 3.2 Experimental Scalability

**MLP Advantages:**
- **Multiple seeds**: Rapid evaluation across different random initializations
- **Privacy budget sweeps**: Fast exploration of ε ∈ {0.5, 1.0, 2.0, 5.0, 10.0}
- **Client configurations**: Efficient FL experiments with varying client counts
- **Hyperparameter tuning**: Quick iteration on learning rates, batch sizes, etc.

**LSTM Limitations:**
- Each experiment takes significantly longer
- Limited ability to run comprehensive parameter sweeps
- Reduced statistical power due to fewer experimental runs

### 3.3 Compatibility with Privacy Techniques

**Differential Privacy:**
- **Per-sample gradients**: MLP's feedforward structure simplifies gradient clipping
- **Noise propagation**: Easier to analyze how DP noise affects each layer
- **Stability**: LayerNorm provides numerical stability without batch dependencies
- **Accounting**: Simpler privacy budget tracking

**Federated Learning:**
- **Communication efficiency**: Smaller model size reduces communication overhead
- **Convergence analysis**: Simpler model enables clearer understanding of FL dynamics
- **Heterogeneity robustness**: Easier to analyze non-IID data distribution effects
- **Aggregation stability**: Simpler gradients reduce aggregation variance

## 4. Trade-off Analysis: Performance vs. Evaluation Precision

### 4.1 Acknowledged Performance Trade-off

We explicitly acknowledge that MLP may achieve lower baseline accuracy than LSTM:

**Expected Performance:**
- **Sleep-EDF**: MLP baseline likely 2-5% lower than LSTM-only (85.53%)
- **WESAD**: MLP baseline likely 3-7% lower than LSTM-only (82.70%)

**Justification:**
- This performance gap is acceptable given our research objectives
- Our goal is to **quantify privacy-accuracy trade-offs**, not maximize absolute accuracy
- Lower baseline makes privacy impact more visible and measurable
- Enables fair comparison: same architecture for baseline, DP, and FL

### 4.2 Research Objectives Alignment

**Primary Research Questions:**
1. How much accuracy degradation occurs with different privacy budgets (ε)?
2. What is the communication overhead of FL for mHealth applications?
3. How do privacy techniques affect different classes (e.g., minority classes)?
4. What are the computational costs of privacy-preserving training?

**MLP Advantages for These Questions:**
- **Question 1**: Cleaner accuracy-ε curves without temporal modeling interference
- **Question 2**: Smaller models enable precise communication cost measurement
- **Question 3**: Simpler model reduces confounding factors in per-class analysis
- **Question 4**: Faster training enables comprehensive computational profiling

### 4.3 Scientific Rigor

**Reproducibility:**
- Faster training enables multiple independent runs
- Better statistical significance through more experimental repetitions
- Reduced variance from simpler model architecture

**Interpretability:**
- Clearer understanding of privacy mechanism effects
- Easier to attribute accuracy changes to specific privacy techniques
- More transparent analysis for scientific publication

## 5. Implementation Details

### 5.1 Unified MLP Model

**Architecture Configuration:**

```python
class UnifiedMLPModel:
    - Input: Features (1D or 2D)
    - Hidden layers: [128, 64] (configurable)
    - Normalization: LayerNorm (DP-compatible)
    - Activation: ReLU
    - Dropout: 0.3 (configurable)
    - Output: n_classes (dataset-specific)
```

**Dataset-Specific Adaptations:**

**Sleep-EDF:**
- Input: 24 features (8 features × 3 channels)
- Output: 5 classes (Wake, N1, N2, N3, REM)
- Hidden: [128, 64]

**WESAD:**
- Input: 140 features (extracted from 14 channels)
- Output: 2 classes (stress, non-stress)
- Hidden: [128, 64]

### 5.2 Temporal Input Handling

For datasets with temporal structure, MLP uses average pooling:

```python
if x.dim() == 3:  # (batch, seq_len, features)
    x = x.mean(dim=1)  # Average over time → (batch, features)
```

This approach:
- Preserves feature information while removing temporal complexity
- Enables unified architecture across different input formats
- Maintains compatibility with feature-extraction preprocessing

## 6. Expected Results and Analysis

### 6.1 Baseline Performance Expectations

**Sleep-EDF:**
- **MLP Baseline**: ~80-83% accuracy (vs. 85.53% LSTM)
- **Acceptable**: 2-5% reduction for improved evaluation precision

**WESAD:**
- **MLP Baseline**: ~75-80% accuracy (vs. 82.70% LSTM)
- **Acceptable**: 3-7% reduction for improved evaluation precision

### 6.2 Privacy Technique Evaluation

**Differential Privacy:**
- **Expected degradation**: Clear, measurable accuracy-ε relationship
- **Analysis**: Precise quantification of privacy-utility trade-off
- **Comparison**: Fair baseline comparison (same architecture)

**Federated Learning:**
- **Convergence**: Faster analysis of FL dynamics
- **Communication**: Accurate overhead measurement
- **Heterogeneity**: Clearer understanding of non-IID effects

### 6.3 Comparative Analysis Benefits

**Same Architecture Across Methods:**
- Baseline, DP, and FL use identical MLP structure
- Eliminates architecture-related confounding factors
- Enables direct attribution of accuracy changes to privacy mechanisms

## 7. Scientific Justification

### 7.1 Research Methodology Alignment

Our choice of MLP aligns with best practices for privacy-preserving ML research:

1. **Controlled Experiments**: Simpler architecture reduces variables
2. **Reproducibility**: Faster training enables more experimental runs
3. **Interpretability**: Clearer understanding of privacy mechanism effects
4. **Scalability**: Enables comprehensive parameter sweeps

### 7.2 Comparison with Related Work

Many privacy-preserving ML studies use simpler architectures (MLP, CNN) rather than complex temporal models (LSTM, Transformer) to:
- Focus on privacy-accuracy trade-offs rather than absolute performance
- Enable extensive experimental evaluation
- Ensure fair comparisons across privacy techniques
- Reduce computational barriers to reproducibility

### 7.3 Practical Implications

**For mHealth Developers:**
- Demonstrates privacy-accuracy trade-offs with realistic, deployable models
- Provides actionable insights for resource-constrained mobile devices
- Shows practical feasibility of privacy-preserving techniques

**For Research Community:**
- Enables reproducible experiments with reasonable computational requirements
- Provides clear baseline for future privacy-preserving mHealth research
- Demonstrates systematic evaluation methodology

## 8. Limitations and Future Work

### 8.1 Acknowledged Limitations

**Performance Trade-off:**
- MLP may not achieve state-of-the-art accuracy
- Temporal information is lost through average pooling
- Feature extraction quality becomes critical

**Scope:**
- Focus on privacy-accuracy trade-offs, not absolute performance optimization
- Results may not generalize to more complex architectures

### 8.2 Future Directions

**Potential Extensions:**
- Evaluate privacy techniques with LSTM for performance comparison
- Develop hybrid approaches combining MLP efficiency with temporal modeling
- Investigate feature extraction methods optimized for MLP privacy analysis

## 9. Conclusions

### 9.1 Decision Summary

**Chosen Architecture: MLP**

**Rationale:**
1. **Evaluation Precision**: Enables precise measurement of privacy-accuracy trade-offs
2. **Experimental Scalability**: Fast training supports comprehensive evaluation
3. **Scientific Rigor**: Cleaner analysis without temporal modeling interference
4. **Practical Relevance**: Demonstrates feasibility for resource-constrained devices

**Trade-off Accepted:**
- Lower baseline accuracy (2-7% reduction) is acceptable
- Research objectives prioritize trade-off analysis over absolute performance
- Same architecture ensures fair comparison across privacy techniques

### 9.2 Impact on Research

This architectural choice enables:

- **Systematic Evaluation**: Comprehensive analysis of privacy-preserving techniques
- **Reproducible Results**: Faster training enables multiple independent runs
- **Clear Insights**: Precise quantification of privacy-accuracy trade-offs
- **Practical Guidelines**: Actionable recommendations for mHealth developers

### 9.3 Scientific Contribution

By choosing MLP over LSTM, we prioritize:
- **Methodological rigor** over absolute performance
- **Evaluation precision** over model complexity
- **Reproducibility** over computational efficiency
- **Scientific clarity** over architectural sophistication

This approach ensures that our privacy-accuracy trade-off analysis is precise, reproducible, and actionable for the mHealth research and development community.

## 10. References

- [1] Dwork, C. (2006). Differential Privacy
- [2] McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data
- [3] Abadi, M. et al. (2016). Deep Learning with Differential Privacy
- [4] WESAD Dataset: Schmidt, P. et al. (2018). Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection
- [5] Sleep-EDF Dataset: Kemp, B. et al. (2000). Analysis of a Sleep-Dependent Neuronal Feedback Loop
