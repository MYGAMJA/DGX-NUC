<!--
source: https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-software/training-environment
raw_md: https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-software/training-environment.md
synced: 2026-05-24
title: 학습환경 (IsaacLab)
-->

# Training Environment

For policy training, we use NVIDIA [Isaac Sim](https://developer.nvidia.com/isaac/sim) and [Isaac Lab](https://developer.nvidia.com/isaac/lab). The code containing Berkeley Humanoid Lite training environment can be installed as an extension to the Isaac Lab.

Some system packages are required by Isaac Lab. We shall install these first.

```bash
sudo apt install cmake build-essential
```

For managing the Python environment, we support using either uv or conda. uv is recommended for its faster speed, modern dependency resolution, and strict reproducibility.

## Setting up environment with uv

[uv](https://docs.astral.sh/uv/) is an extremely fast Python package and project manager.

To set up uv, run this command:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Then, simply run this command to install everything

```bash
uv sync
```

Finally, to activate the environment, do

```bash
source ./.venv/bin/activate
```

## Setting up environment with conda

Alternatively, [conda](https://anaconda.org/anaconda/conda) can be used to manage Python packages.

First, create a conda environment to contain all the Python packages.

```bash
conda create -yn berkeley-humanoid-lite python=3.10
conda activate berkeley-humanoid-lite
```

### Installing Isaac Sim and Isaac Lab

Please refer to [Issac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) for more detailed guidance on installing Isaac Sim and Isaac Lab.

Here, we provide the procedure we followed to install on our machines. We have verified this flow on a fresh install of Ubuntu 24.04.

Use this command to install Isaac Sim 4.5.0 and Isaac Lab 2.1.0.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install isaacsim[all,extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com
pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com
```

### Setting up Berkeley-Humanoid-Lite extension

After installing the Isaac Lab environment, we can proceed to install the Berkeley Humanoid Lite Extension.

We start by cloning the project repository.

```bash
git clone https://github.com/HybridRobotics/Berkeley-Humanoid-Lite.git
```

The project repository contains submodules for robot description and low-level code. The submodules can be initialize with this command.

```bash
cd Berkeley-Humanoid-Lite
git submodule update --init
```

Then, install the modules to IsaacLab environment.

```bash
pip install -e ./source/berkeley_humanoid_lite/
pip install -e ./source/berkeley_humanoid_lite_assets/
pip install -e ./source/berkeley_humanoid_lite_lowlevel/
```

There are some additional packages required by our project. Run the following script to install them.

```bash
pip install -r requirements.txt
```

## Training the policy

Two tasks are defined in the codebase. `Velocity-Berkeley-Humanoid-Lite-v0` trains a policy that actuates all 22 degrees of freedom on the robot, and `Velocity-Berkeley-Humanoid-Lite-Biped-v0`  trains a policy that only controls the 12 degrees of freedom on the leg.&#x20;

To train the policy, run this following command. By default, we train for 6000 iterations, which should take around two hours.

{% tabs %}
{% tab title="uv" %}
{% code overflow="wrap" %}

```bash
uv run ./scripts/rsl_rl/train.py --task Velocity-Berkeley-Humanoid-Lite-v0 --headless
```

{% endcode %}
{% endtab %}

{% tab title="conda" %}
{% code overflow="wrap" %}

```bash
python ./scripts/rsl_rl/train.py --task Velocity-Berkeley-Humanoid-Lite-v0 --headless
```

{% endcode %}
{% endtab %}
{% endtabs %}

To view the trained result, run the following command.

{% tabs %}
{% tab title="uv" %}
{% code overflow="wrap" %}

```bash
uv run ./scripts/rsl_rl/play.py --task Velocity-Berkeley-Humanoid-Lite-v0 --num_envs 16
```

{% endcode %}
{% endtab %}

{% tab title="conda" %}
{% code overflow="wrap" %}

```bash
python ./scripts/rsl_rl/play.py --task Velocity-Berkeley-Humanoid-Lite-v0 --num_envs 16
```

{% endcode %}
{% endtab %}
{% endtabs %}

In addition to get a visualization of the policy, the `play.py` script will also export the trained policy into an ONNX file for deployment. It will also create or update the `configs/policy_latest.yaml` configuration file, which records the parameters that will be used during sim2sim and sim2real deployment.

We will see how to use this exported model checkpoint and configuration file in the next section.


---
