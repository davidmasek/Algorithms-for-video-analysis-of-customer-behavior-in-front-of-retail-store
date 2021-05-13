# Algorithms for video analysis of customer behavior in front of retail store

This repository contains the source code and suplementary materials for my bachelor thesis at [FIT CTU](https://fit.cvut.cz/).

## Content

The `thesis` directory contains the latex sources for the thesis and the generated PDF file.

The main algorithm pipeline source code is not included, because it is part of a cooperation project between FIT CTU and industrial sphere. The collected dataset is not included for privacy reasons.

The repository contains Jupyter notebooks with some of the experiments from the thesis, specifically:

- `evaluate_age.ipynb` - evaluation of the *SSRNet* model on the *MegaAge* dataset, with focus on difference after optimizing with TensorRT framework,
- `perfromance.ipynb` - evaluation of the performance gain from using the TensorRT framework,
- `tracker_summary.ipynb` - overview of the algorithm's results,
- `face_alignment.ipynb` - initial experiments with a *face alignment* model on samples from the *LS3D-W* dataset.

The notebooks are supported by Python source code. The main modules are:

- `trt_infernece.py` - includes classes for building TensorRT engines from ONNX models and using them for inference
- `onnx_inference.py` - includes wrapper around the [ONNX Runtime](https://www.onnxruntime.ai/) library
- `timing.py` - includes utilities for timing Python code


## Requirements

See `requirements.txt` for specific libraries and versions. In general you need:
- [ONNX Runtime](https://www.onnxruntime.ai) for inference with ONNX models,
- [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) for using the TensorRT framework, note that this also requires supported NVIDIA graphics card and has complex dependencies,
- [face-alignment](https://github.com/1adrianb/face-alignment) for the face alignment experiments,
- common Python libraries for data processing such as [NumPy](https://numpy.org/).

## Credits

See the thesis and the individual files for details. Thesis citation:

> Mašek, David. Algorithms for video analysis of customer behavior in front of retail store.  Bachelor's thesis.  Czech Technical University in Prague, Faculty of Information Technology, 2021.