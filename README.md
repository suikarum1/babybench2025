# BabyBench 2025 Competition Starter Kit

BabyBench is a multimodal benchmark for intrinsic motivations and open-ended learning in developmental artificial ingelligence. The objective is to teach typical behavioral milestones to [MIMo](https://github.com/trieschlab/MIMo), a multimodal infant model. We provide the embodiment, the simulation environments, and the evaluation metrics; all you need is to implement your ideas.

The first **BabyBench Competition** will take place at the [***IEEE* ICDL 2025 Conference**](https://icdl2025.fel.cvut.cz/) in Prague! The topic of this first edition will be *how infants discover their own bodies*. Can you help MIMo learn two typical infant behaviors: **self touch** and **hand regard**?  Make your submission and know more about the competition [here](https://babybench.github.io/2025).

## Index

* [Install the software](#install-the-software)
* [Getting started](#getting-started)
* [Prepare your submission](#prepare-your-submission)
* [How to...?](#how-to)
* [Support](#support)

## Install the software

### Option 1: Local installation

Pre-requisites: [Python](https://www.python.org/), [Git](https://git-scm.com/), and [Conda](https://www.anaconda.com/products/individual). Tested on Ubuntu 18.04 and 24.04, MS Windows 10 and 11, and MacOS 15.

1. Create a conda environment:
   
   ```
   conda create --name babybench python=3.12
   conda activate babybench
   ```

2. Clone this repository: 
   
   ```
   git clone https://github.com/babybench/BabyBench2025_Starter_Kit.git
   cd BabyBench2025_Starter_Kit
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

This will run a test to check that the everything is correctly installed. If the installation is successful, you should find a new folder called `test_installation` in the `results` directory with a rendered video of the test.

### Option 2: Singularity container

Pre-requisites: [Singularity](https://neuro.debian.net/install_pkg.html?p=singularity-container). Tested on Ubuntu 24.04.

1. Create the singularity container

```
singularity build -F babybench.sif babybench.def
```

This will create a singularity container called `babybench.sif` in the current directory.

2. Launch the container

```
singularity run -c -H /home --bind "$PWD/:/home" babybench.sif
```

This will run a test to check that the everything is correctly installed. If the installation is successful, you should find a new folder called `test_installation` in the `results` directory with a rendered video of the test.

### Troubleshooting

If you encounter any issues, visit the [troubleshooting page](https://babybench.github.io/2025/troubleshooting)

## Getting started

If you are not sure where to begin, we recommend having a look at the `examples` directory and [this wiki page](https://babybench.github.io/2025/start).

The aim for BabyBench is to get MIMo to learn the [target behaviors](https://babybench.github.io/2025/about) without any external supervision, i.e. without extrinsic rewards. Your goal is to train a policy that matches sensory observation (proprioception, vision, touch) to actions. To do so, we provide an [API](https://babybench.github.io/2025/API) to initialize and interact with the environments.


## Prepare your submission

Submissions must be made through [PaperPlaza](https://ras.papercept.net/). The topic of this first BabyBench competition is *how infants discover their own bodies*. There are two target behaviors: **self-touch** and **hand regard**.

First-round submission should consist of:

- A 2-page extended abstract explaining your method,

- The training log file generated during training,

- The trajectory log files generated during the evaluation,

- *Optionally*, you can also submit links to the following:  

   * A repository with your training code,
   * A folder with videos of the learned behaviors rendered during the evaluation.

## How to...?

For further information, check our [Wiki](https://github.com/babybench/2025).  
In particular, if you want to know more about:

- the training environments, see [here](https://babybench.github.io/2025/API)
- the target behaviors, see [here](https://babybench.github.io/2025/about)
- examples, see [here](https://babybench.github.io/2025/start)
- how to generate the submission files, see [here](https://babybench.github.io/2025/submission)
- the evaluation process, see [here](https://babybench.github.io/2025/competition)
- resources about intrinsic motivations and open-ended learning, see [here](https://babybench.github.io/2025/start)

... or see the [FAQ](https://babybench.github.io/2025/faq) for common questions or errors.

## Support

Feel free to contact us for any questions about BabyBench. You can post an issue [here](https://github.com/babybench/BabyBench2025_Starter_Kit/issues) on Github or [contact the organizers via mail](mailto:fcomlop@gmail.com?subject=[BabyBench]%20Question).

We highly encourage you to collaborate with other participants! You can submit your problems, questions or ideas in the [discussion forum](https://github.com/babybench/BabyBench2025_Starter_Kit/discussions).   

## License

This project is licensed under an MIT License - see [LICENSE](https://github.com/babybench/BabyBench2025_Starter_Kit/blob/main/LICENSE) for details
