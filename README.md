# DUPLEX
Duplication-crossbar exploration for on-chip random access. The DUPLEX architecture combines bank duplication and dynamic crossbar routing to allow parallel reads under random accessing for on-chip memory. The DSE selects the optimal configuration under a performance or resource target.

## Requirements
- Python ≥ 3.9 and `pip install -r requirements.txt`. 

- Hardware build only: Vitis 2024.2, VCK190 base platform, Versal common image.

## Repository Layout 

```
duplex/
├── dse/                Design-space exploration: simulator, resource model, DSE scripts
├── workloads/          Read-trace generators (uniform, Zipf, perspective), SpMV matrices
├── hw/                 HLS kernel, host code, and Makefile for the VCK190
├── requirements.txt
└── LICENSE
```