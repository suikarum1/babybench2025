### BabyBench 2025
##### Exploration by Disagreement in an Embodied Infant: Self-Touch and Hand Regard
We investigate emergent sensorimotor devel- opment in a simulated infant (MIMo) through disagreement-driven exploration. Our method uses an ensemble of linear forward models to predict compressed sensory features, with ensemble variance serving as an intrinsic reward signal that drives exploration toward novel but learnable sensorimotor configurations. Our framework provides a com- putationally efficient, modality-agnostic approach to intrinsically motivated sensorimotor learning that captures key aspects of early infant development.

After local installation of BabyBench Starter Kit (https://github.com/babybench/babybench2025_starter_kit), the disagreement-driven exploration code can be ran and evaluated as following, taking the self-touch task as an example:
1. Model training:
   
   ```
   python ./examples/intrinsic_selftouch_disagreement.py --train_for=500000                                        
   ```
2. Results and Evaluation:
   
   ```
   python evaluation_selftouch_disagreement.py --config=examples/config_selftouch.yml --episodes=5 --duration=10000                                    
   ``` 
This step generates preview videos of the prediction on infant behaviors.