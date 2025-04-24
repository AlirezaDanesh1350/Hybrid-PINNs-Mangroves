README: Reproducibility Package for Hybrid PINNs Framework

This archive contains the essential files to reproduce the results presented in the manuscript titled:
“A Hybrid Physics-Informed Neural Network Framework for Modelling Hydro-Morphodynamic Processes in Mangrove Environments”

Contents
--------
- original_with_physics.py
  → Main script implementing the hybrid PINNs model using NVIDIA Modulus.
- morpho_equation.py
  → Supporting script containing the sediment transport equation and associated physics constraints.
- original_simplified.csv
  → Input dataset used to train and evaluate the hybrid PINNs model.

System Requirements
-------------------
- NVIDIA GPU with CUDA support
- Docker installed and configured
- NVIDIA Modulus (physics-informed deep learning framework)

  ℹ️ NVIDIA Modulus can be run using Docker containers following the official installation guide:
  https://developer.nvidia.com/modulus

Quick Start Instructions
------------------------
1. Install Docker and NVIDIA Modulus
   Follow the official Modulus setup instructions:
   https://docs.nvidia.com/modulus/latest/index.html

2. Launch the Docker container
   Make sure you mount this extracted folder into the container (e.g., using the `-v` option).

3. Run the model
   Inside the container:
   python original_with_physics.py

4. Edit and experiment
   You can modify morpho_equation.py to test different PDE formulations or boundary conditions.

Contact
-------
For questions or technical support, please contact the corresponding author of the manuscript.
