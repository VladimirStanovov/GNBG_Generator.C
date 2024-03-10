# GNBG_Generator.C
C++ implementation of Generalized Numerical Benchmark Generator (GNBG)

Includes implementation of simple Differential Evolution (DE) with rand/1 strategy and binomial crossover

Problem parameters can be saved to file for further usage

Competition page: https://competition-hub.github.io/GNBG-Competition/

Reference:

D. Yazdani, M. N. Omidvar, D. Yazdani, K. Deb, and A. H. Gandomi, "GNBG: A Generalized
  and Configurable Benchmark Generator for Continuous Numerical Optimization," arXiv prepring	arXiv:2312.07083, 2023.
  
A. H. Gandomi, D. Yazdani, M. N. Omidvar, and K. Deb, "GNBG-Generated Test Suite for Box-Constrained Numerical Global
  Optimization," arXiv preprint arXiv:2312.07034, 2023.
  
MATLAB version: https://github.com/Danial-Yazdani/GNBG_Generator.MATLAB

Python version: https://github.com/Danial-Yazdani/GNBG_Generator.Python

# Compilation and usage

Compile the gnbg-generator-c++.cpp file with any compiler (e.g. GCC):

g++ -std=c++11 -O3 gnbg-generator-c++.cpp -o gnbg-generator-c++

There are three operation modes:

0: Creates file "func.txt" containing the parameters of the generated test function, also tests random points.

1: Creates file "func.txt" containing the parameters of the generated test function, generates grid and evaluates the function, saves for visualization in "vis.txt". You may then run visualization code with "python vis.py".

2: Creates file "func.txt" containing the parameters of the generated test function, runs differential evolution algorithm on this function and saves results to "Res_DE_.txt".
