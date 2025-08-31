## BabyBench 2025
### Exploration by Disagreement in an Embodied Infant: Self-Touch and Hand Regard

This repository contains the code and configuration files for our BabyBench 2025 extended abstract submission to ICDL 2025.

We investigate emergent sensorimotor development in a simulated infant (MIMo: https://github.com/trieschlab/MIMo) using a disagreement-driven intrinsic motivation strategy. Our method trains an ensemble of lightweight linear forward models to predict compressed sensory features. The ensemble variance serves as an intrinsic reward signal, encouraging the agent to explore novel yet learnable sensorimotor configurations.

This framework is:
- Computationally efficient
- Modality-agnostic
- Capable of capturing early developmental phenomena such as self-touch and hand regard

After local installation of BabyBench 2025 Starter Kit (https://github.com/babybench/babybench2025_starter_kit), you can run and evaluate our disagreement-based exploration method on different behavioral tasks as follows, taking the self-touch task as an example:
1. Model training:
   
   ```
   python ./examples/intrinsic_selftouch_disagreement.py --train_for=500000                                        
   ```
2. Results and Evaluation:
   
   ```
   python evaluation_selftouch_disagreement.py --config=examples/config_selftouch.yml --episodes=5 --duration=10000                                    
   ``` 
This step will generate preview videos of the infant's behavior under the learned intrinsic policies.
