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

## Design-space exploration (`dse/`)

The DSE picks a design configuration (M, N_k, N_w, D) for a given workload. The cycle-accurate simulator models on bank group and the resource model estimates BRAM, FF, and LUT utilization. The DSE either finds the optimal configuration under a performance target, trying to minimize resource utilization, or under a resource target, trying to maximize throughput in reads/cycle. It then compares the chosen configuration against full duplication and full crossbar baseline results.

## Workloads (`workloads/`)

Four workloads are used in DUPLEX's evaluation: uniform random reads, Zipf-distributed reads, with skew values of 0.5, 1.0, and 1.5, SpMV accesses with Gini indices of 0.35, 0.51, and 0.70, and perspective transformation indices from a real-world matrix, a 30 degree rotation, and a zoom out. 

## Hardware (`hw/`)

The hardware directory contains the HLS implementation of DUPLEX created for the AMD Versal VCK190, built with Vitis 2024.2. The configuration is set through the Makefile which can change M, Nk, Nw, and D parameters. The Makefile compiles, links, and packages the build for an SD card image that can be run on board with the host application.