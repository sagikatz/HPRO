# HPRO: Direct Visibility of Point Clouds for Optimization
**Authors**: Sagi Katz, Ayellet Tal

## Overview

**HPRO** introduces a novel, differentiable approximation method for determining the visibility of points in a point cloud from a given viewpoint. The method overcomes the limitations of existing approximation techniques, which are not suitable for optimization processes or learning models due to their lack of differentiability. 

This approach leverages the concept of extreme points of a point set, enabling efficient visibility computation in a way that integrates seamlessly into optimization algorithms or neural network layers. It is especially useful for tasks such as optimal viewpoint selection.

## Key Features

- **Differentiable Visibility Computation**: A smooth, differentiable approach to determining which points of a point cloud are visible from a given viewpoint.
- **Optimization-Friendly**: The method is well-suited for optimization tasks, enabling its direct use in gradient-based algorithms.
- **Neural Network Integration**: The method can be incorporated as a layer in neural networks, facilitating its application in learning models.
- **Theoretical Foundation**: The correctness of the operator is rigorously proven in the limit, ensuring its reliability in practical use cases.

## Theoretical Background

The HPRO method is grounded in the idea that a point cloud is a sampling of a continuous surface, and the problem of determining visible points can be thought of as an optimization of visibility in a differentiable manner. The method identifies extreme points within a point set in a smooth and computationally efficient way.

The paper provides rigorous proofs of the correctness of the operator in the limit, which further validates its use in various practical applications.

## Installation

To use the HPRO method in your project, you can clone the repository and install the required dependencies.

### Clone the repository:
```bash
git clone https://github.com/sagikatz/hpro.git
cd hpro
```

### Install dependencies:
```bash
pip install -r requirements.txt
```
## Example

Run demo.py for an example that compares HPRO and HPR to the ground truth for a provided model.


## License

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
