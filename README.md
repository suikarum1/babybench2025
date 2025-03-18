# BabyBench 2025 Competition Starter Kit

BabyBench is a benchmark for intrinsic motivations and open-ended learning in developmental artificial ingelligence. The objective is to teach typical behavioral milestones to [MIMo](https://github.com/trieschlab/MIMo), a multimodal infant model. We provide the embodiment, the simulation environments, and the evaluation metrics; all you need is to implement your ideas.

We are holding the first **BabyBench Competition at the *IEEE* ICDL 2025 Conference** in Prague! The topic of this first edition is *the emergence of the self*. Can you train MIMo to develop **self-touch**, **hand regard**, or **mirror recognition**? Make your submission and know more about the competition [here](https://icdl2025.fel.cvut.cz/baby-bench/).

## Index

* [Install the software](#install-the-software)
* [Train MIMo](#train-mimo)
* [Make your submission](#make-your-submission)
* [How to...?](#how-to)
* [Support](#support)

## Install the software

Pre-requisites: [Python](https://www.python.org/), [Git](https://git-scm.com/), and [Conda](https://www.anaconda.com/products/individual). All software has been tested on Ubuntu 18.04 and 24.04.  

1. Create a conda environment:
   
   ```
   conda env create --name BabyBench python=3.12
   conda activate BabyBench
   ```

2. Clone this repository: 
   
   ```
   git clone https://github.com/fcomlop/BabyBench.git
   cd BabyBench
   ```

3. Install requirements:
   
   ```
   pip install -r requirements.txt
   ```

4. Clone and install MIMo:
   
   ```
   pip install -e MIMo
   ```

All done! You are ready to start using BabyBench. 

5. Launch the installation test:  
   
   ```
   python test_installation.py
   ```

This will run a test to check that the everything is correctly installed. If you encounter any issues, visit the [troubleshooting page](https://github.com/fcomlop/BabyBench/wiki/troubleshooting)

## Train MIMo

The aim for BabyBench is to get MIMo to learn one or more of the [target behaviors](https://github.com/fcomlop/wiki/environments) without any external supervision, i.e. without extrinsic rewards. Your goal is to train a policy that matches sensory observation (proprioception, vision, touch) to actions. To do so, we provide some [environments](https://github.com/fcomlop/wiki/environments). To change the configuration, simply change the values of `config.yml`.

If you are not sure where to begin, we recommend having a look at the `examples` directory and [this wiki page](https://github.com/fcomlop/wiki/examples).

## Make your submission

Submissions must be made through the [ICDL 2025 BabyBench Competition website](https://icdl2025.fel.cvut.cz/baby-bench/). The topic of this first BabyBench competition is *the emergence of the self*. There are three target behaviors: **self-touch**, **hand regard**, and **mirror recognition**. 

First-round submission should consist of:

* A 2-page abstract explaining your method,

* The videos and logs generated during training,

* The training code (*optional, only required to verify the winner of the competition*).

## How to...?

For further information, check our [Wiki](https://github.com/fcomlop/BabyBench/wiki).  
In particular, if you want to know more about:

- the training environments, see [here](https://github.com/fcomlop/wiki/environments)
- the target behaviors, see [here](https://github.com/fcomlop/wiki/behaviors)
- examples, see [here](https://github.com/fcomlop/wiki/examples)
- how to generate the submission files, see [here](https://github.com/fcomlop/wiki/submission)
- the evaluation process, see [here](https://github.com/fcomlop/wiki/evaluation)
- resources about intrinsic motivations and open-ended learning, see [here](https://github.com/fcomlop/wiki/resources)

... or see the [FAQ](https://github.com/fcomlop/wiki/FAQ) for common questions or errors.

## Support

Feel free to contact us for any We thank the developers of the REAL cmopetition for problems or additional questions about BabyBench. You can post an issue [here](https://github.com/fcomlop/BabyBench/issues) on Github or directly [contact the organizers via mail](mailto:fcomlop@gmail.com?subject=[BabyBench]%20Question).

We highly encourage you to collaborate with other participants! You can submit your problems, questions or ideas in the [discussion forum](https://github.com/fcomlop/BabyBench/discussions).   

## Acknowledgements

## License

This project is licensed under an MIT License - see [LICENSE](https://github.com/fcomlop/MIMo/LICENSE) for details
