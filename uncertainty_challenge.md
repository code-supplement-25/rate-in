# What Comes Next?

To build intuition about uncertainty, consider this short puzzle:

**What is the next number in the sequence?**  
**4, 5, 14, 15, …**  
**Is it 24 or 40?**  
Both are reasonable answers.

Some may identify a repeating pattern in the differences: **1, 9, 1, 9**, leading to **24**.  
Others might notice that these are **successive numbers that begin with the letter 'F'**, suggesting the next in line is **forty**.

Despite observing the same data, we arrive at **different hypotheses** about the underlying rule.

This reflects a state of **uncertainty** — when multiple plausible explanations exist, but the data alone does not dictate a unique answer.

---

## The Same Challenge in Machine Learning

This dilemma of multiple plausible rules hiding beneath the same observations also arises in machine learning models.

As models have grown in complexity and size, it has become infeasible to represent uncertainty over all possible configurations.  
Training typically commits to a single set of parameters, the one that best fits the data, locking in one explanation among many.

At inference time, this choice remains fixed. The model returns the same output for a given input, with no indication of alternative plausible outcomes.

Quantifying this uncertainty is essential for building models that are not only accurate but also reliable, especially in domains where multiple interpretations may be valid.

---

## Recovering Uncertainty After Training

So how do we recover other plausible hypotheses, after the model has already been trained?

One practical approach is to use **dropout at inference time**, activating different parts of the network on each forward pass.  
Each variation reflects a different plausible hypothesis, allowing us to sample from a distribution of predictions.

But not all dropout is equal.

Standard Monte Carlo Dropout typically applies a fixed dropout rate across all layers and inputs.  
This can lead to suboptimal behavior — dropping too much where features are important, and too little where redundancy could be leveraged.  
The result: noisy, poorly calibrated, or spatially misaligned uncertainty estimates.

---

## Introducing Rate-In

**Rate-In** is a post-training, unsupervised method that adjusts dropout rates per input and per layer in any trained network.

It follows a few simple steps:

1. **Apply dropout** to layer activations  
2. **Estimate information loss** (e.g., via mutual information)  
3. **Adjust rates** to meet target loss thresholds

---

### **Rate-In's Core Principle:**

→ Retain more when features matter  
→ Drop more when they don't  
Dropout becomes **adaptive noise, driven by information**

---

**To learn more, read our paper:**  
**_Rate-In: Information-Driven Adaptive Dropout Rates for Improved Inference-Time Uncertainty Estimation_**  
*Tal Zeevi, Ravid Shwartz-Ziv, Yann LeCun, Lawrence H. Staib, John A. Onofrey*  
[arXiv:2412.07169](https://arxiv.org/abs/2412.07169)

Or keep exploring this [GitHub repo](https://github.com/code-supplement-25/rate-in/tree/main).
