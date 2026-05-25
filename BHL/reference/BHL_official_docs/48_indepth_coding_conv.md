<!--
source: https://berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/training-environment-coding-convention
raw_md: https://berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/training-environment-coding-convention.md
synced: 2026-05-24
title: 학습 코딩 컨벤션
-->

# Training Environment Coding Convention

We found that keeping track of all the reward functions, configuration settings, and robot parameters in the training environment can quickly become overwhelming. To help manage this complexity, we’ve adopted a coding convention that organizes these elements consistently, making them easier to locate and maintain.

### Environment configuration class ordering

The configuration classes within a environment definition is ordered as follows:

```python
@configclass
class YourEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for your robot learning environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg()

    # Policy commands
    commands: CommandsCfg = CommandsCfg()

    # Policy observations
    observations: ObservationsCfg = ObservationsCfg()

    # Policy actions
    actions: ActionsCfg = ActionsCfg()

    # Policy rewards
    rewards: RewardsCfg = RewardsCfg()

    # Termination conditions
    terminations: TerminationsCfg = TerminationsCfg()

    # Randomization events
    events: EventsCfg = EventsCfg()

    # Curriculum
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # override default configuration parameters

```

### Configuration term parameter ordering

For the parameter within each term, we order the parameters based on the class argument ordering:

```python
# example observation term
base_ang_vel = ObsTerm(
    func=mdp.base_ang_vel,
    noise=Unoise(n_min=-0.2, n_max=0.2),
    scale=0.2,
)
# example reward term
track_lin_vel_xy_exp = RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    weight=1.0,
)
```

### Asymmetric observation config

Typically, for asymmetric observations, we will add all actor observations to the critic along with some additional privileged observations. To ensure that the configurations of the duplicated terms are consistent, the critic observation config should inherit from the actor's.

One thing to note is that we do not want to add noise to the critic's observation terms. To do this, we can set the `enable_corruption` parameter to `False` .

The resulting code should look like

```python
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.2,
        )

        # ... (more terms)

        def __post_init__(self):
            self.enable_corruption = True  # <-- enable noise for actor obs

    class CriticCfg(PolicyCfg):  # <-- inherit all terms from the actor
        """Observations for critic group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        # ... (more terms)

        def __post_init__(self):
            self.enable_corruption = False  # <-- disable noise for critic obs

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

```

### Reward terms

Reward terms are grouped by their purpose. Measurements on task-space performance is put at the front due to its importance. Then, terms for basic behaviors, such as survival, motion smoothness etc. are followed. Lastly, terms for "fine-tuning" the policy are added.

```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # === Reward for task-space performance ===
    # ... (terms)

    # === Reward for basic behaviors ===
    # ... (terms)

    # === Reward for encouraging behaviors ===
    # ... (terms)

```

### \_\_post\_init\_\_ behavior

Differ from the Isaac Lab code, we encourage to set configurations in the corresponding config class and config terms. Only override parameters in the `__post_init__()` function:

1. during temporary debugging and parameter space explorations,
2. when the base configuration is inaccessible (for example, the `ROBOT_CFG` variable)


---
