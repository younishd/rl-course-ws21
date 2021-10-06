# Course: Introduction to Reinforcement Learning

The recommended way to setup your development environment is to use Anaconda:
1. Download and install Miniconda for your OS from here: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

2. Start the conda terminal and create a new environment for this course:

`conda create --name rl-course python=3.8`

3. Activate this environment and install OpenAi Gym inside the new env with pip3:

`conda activate rl-course`

4. Install the `numpy` and `gym` packages

`conda install -c conda-forge gym`

4. Test your setup by running:

`python3 1_FrozenLake_Random.py`

