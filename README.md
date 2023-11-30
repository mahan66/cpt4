# Continual Prompted Transformer for Test-Time Training (CPT4)

Adapting machine learning models to non-stationary environments presents a significant challenge due to the evolving domain shifts over time. Test-time adaptation (TTA) methods aim to address this challenge by leveraging pre-trained models on source data to make predictions on diverse target data domains, and adapting the model using unlabeled target data. However, real-world scenarios often involve continual shifts in target data domains during testing, introducing complexities related to ongoing adaptation and the potential propagation of errors.

While earlier TTA methods were primarily focused on convolutional-based models, this repository introduces an innovative transformer-based approach to tackle the challenges associated with TTA, especially in non-stationary environments. We present a novel method, named Continual Prompted Transformer for Test-Time Training (CPT4), designed to enhance the Vision Transformer (ViT) model. CPT4 incorporates shared prompts (small learnable parameters) and a batch normalization module, aiming to mitigate catastrophic forgetting and effectively handle domain shifts.

<p align="center">
<img src="https://github.com/mahan66/cpt4/blob/main/cpt4_prompted_batchNormed_vit.png" alt="CPT4 Block Diagram" width="80%"/>
<p>

## Key Features

- **Shared Prompts:** The introduction of a prompt pool retains information from prior tasks, facilitating continual learning during test time.
  
- **Batch Normalization Module:** This module transfers source data statistics to test time, contributing to the model's adaptability.

## Methodology

Our approach is grounded in continual learning for TTA scenarios without access to source data or target labels. We conducted a comprehensive set of experiments on various popular continual image classification benchmarks featuring non-stationary environments during test time.

## Results

The experimental results showcase that CPT4 consistently outperforms the original ViT model across different adaptation strategies. This work contributes to the ongoing exploration of utilizing small learnable parameters for continual learning in TTA scenarios.

## How to Use

For details on implementing and utilizing CPT4, please refer to the documentation and code provided in this repository.

## Citation

If you find our work helpful, consider citing our paper:

[CPT4: Continual Prompted Transformer for Test Time Training]

## License

This project is licensed under the [MIT].
