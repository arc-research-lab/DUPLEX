# DUPLEX
Duplication-crossbar exploration for on-chip random access. The DUPLEX architecture combines bank duplication and dynamic crossbar routing to allow parallel reads under random accessing for on-chip memory. The DSE selects the optimal configuration under a performance or resource target.

## Requirements
- requirements.txt
- AMD Vitis 2024.2 
- VCK190 base platform
- Versal common image 

## Repository Layout 
1. DSE directory: contains design space exploration code, including performance and resource models and DSE scripts
2. HW directory: contains HLS kernel code, host code, and a Makefile 
3. Workloads directory: contains generation code for uniform, Zipf, and perspective transforms, as well as sparse matrix-vector matrices used in the paper