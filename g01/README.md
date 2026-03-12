# Assignment 01 - Image Warping

This repository folder is an implementation of DIP-Teaching Assignment 01 (Image Warping).

## Requirements

To install requirements:

```bash
python -m pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`.

## Training

This assignment does not include model training. Instead, you can generate deterministic visual artifacts with:

```bash
python generate_results.py
```

## Evaluation

To run interactive evaluation for global geometric transformation:

```bash
python run_global_transform.py
```

To run interactive evaluation for point-guided deformation:

```bash
python run_point_transform.py
```

To run a quick non-interactive verification:

```bash
python -m py_compile run_global_transform.py run_point_transform.py generate_results.py
python -c "import run_global_transform as g; import run_point_transform as p; import generate_results as r; print('imports_ok')"
```

## Pre-trained Models

Not applicable for this assignment.

## Results

Run:

```bash
python generate_results.py
```

Generated files are saved in `results/`.

| Task | Artifact | Reproduction Command |
| --- | --- | --- |
| Global transform | `results/global_input.png` | `python generate_results.py` |
| Global transform | `results/global_scale_1p4.png` | `python generate_results.py` |
| Global transform | `results/global_rotate_35.png` | `python generate_results.py` |
| Global transform | `results/global_translate.png` | `python generate_results.py` |
| Global transform | `results/global_flip.png` | `python generate_results.py` |
| Global transform | `results/global_combo.png` | `python generate_results.py` |
| Point deformation | `results/point_controls.png` | `python generate_results.py` |
| Point deformation | `results/point_warped.png` | `python generate_results.py` |
| Point deformation | `results/point_before_after.png` | `python generate_results.py` |

## Contributing

This folder is coursework submission content. For updates, keep changes reproducible and include:

1. implementation files,
2. commands to reproduce outputs,
3. updated result artifacts and notes in `README_submission.md`.
