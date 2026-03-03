# EE490 Project 1 — PCA/SVD Image Compression & Denoising + AdaDelta Optimization

**Course:** EE 490 – Engineering Applications of AI and Machine Learning  
**Instructor:** Dr. Jianwu Zeng  
**Author:** Nathnael Minuta  
**Date:** Feb 20, 2026

## Overview
This project has two parts:

### Part 1 — Image Compression & Denoising using PCA/SVD
- Apply SVD to each RGB channel of an image
- Reconstruct using rank-k approximation
- Use truncated singular values for:
  - **Compression** (reduce storage)
  - **Denoising** (suppress noise)

### Part 2 — Optimization using Gradient Descent (AdaDelta-style)
- Minimize the quadratic objective:
  \[
  f(x,y) = x^2 + 2xy + 4y^2
  \]
- Use an AdaDelta-style adaptive step size to converge to the global minimum at (0,0)
- Visualize:
  - Trajectory on the objective surface
  - Effective learning rate dynamics

## Repository Layout
- `src/` — MATLAB scripts
- `data/` — input images (or placeholders if not included)
- `results/` — exported plots, tables, and screenshots
- `report/` — final report PDF

## How to Run (MATLAB)
1. Open MATLAB and set the repo root as the working directory
2. Run:
   - `src/Project1_SVD_Denoising.m`
   - `src/Project1_SVD_Compression.m`
   - `src/Project1_Gradient_Descent.m`

## Notes
- If you do not want to upload images publicly, keep `data/` empty and add your own images locally.
- Results in `results/figures/` are exported from MATLAB for documentation.

## Key Results (from report)
- Denoising best tradeoff: **k = 60**
- Compression best tradeoff: **k = 80**
- AdaDelta-style optimizer converged in ~17 iterations (example run)

## License
MIT License (see `LICENSE`).
