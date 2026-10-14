## SGDTM: Structure-Aware Global-Local Diffusion Transformer Mamba for Multimodal Remote Sensing Image Classification

## Abstract

Multimodal remote sensing image classification is of critical importance for land-cover observation and interpretation. Current mainstream methods, such as CNNs, and Transformers, can extract multimodal complementary information, but their limited local receptive fields, or high computational complexity restrict classification accuracy and efficiency in complex scenes. The emerging state space model Mamba offers a linear-complexity advantage, yet its fixed raster scanning has a structural conflict with the irregular land-cover distribution, spatial anisotropy, and modality heterogeneity inherent in remote sensing scenes, which easily leads to the disruption of spatial continuity and the weakening of local priors. In addition, statistical distribution discrepancies among heterogeneous modalities further increase the difficulty of joint modeling. To address these limitations, we propose SGDTM, a structure-aware global-local diffusion Transformer Mamba model that unifies structure-aware sequence organization and global-local context modeling within a single end-to-end framework. SGDTM coordinates two modules: SASSM learns an implicit structural partition via prototype assignment, introduces reversible reordering to enable state space scanning to propagate along structure-consistent sequences, and then produces structure-enhanced features through dual-view Transformer encoding, gated fusion, and channel attention; GL-DiT employs linear global attention and depthwise convolutional local enhancement to perform global-local aggregation of structure-enhanced features within a single DiT block. This joint design effectively alleviates the order sensitivity of Mamba and its insufficient local priors. Extensive experiments on datasets such as Muulf, Houston, Augsburg, Pavia University, and MPOLSAR show that SGDTM achieves the highest accuracy across all classification metrics, significantly outperforming existing representative methods. In addition, SGDTM exhibits strong generalization ability across heterogeneous modalities such as hyperspectral and PolSAR, validating its broad applicability to multimodal remote sensing classification tasks. 

## Requirements:

- Python 3.11
- PyTorch >= 2.10.0
- mamba-ssm >= 2.3.0

## Usage:

python main.py

