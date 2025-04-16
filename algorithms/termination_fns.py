"""
The world models we use in model-based RL don't predict termination, so we need to
define termination functions for each task.
This code is adapted from
https://github.com/yihaosun1124/OfflineRL-Kit/blob/6e578d13568fa934096baa2ca96e38e1fa44a233/offlinerlkit/utils/termination_fns.py#L123
Thanks to the authors!
"""

import jax.numpy as jnp


def termination_fn_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    not_done = jnp.logical_and(jnp.all(next_obs > -100), jnp.all(next_obs < 100))
    done = ~not_done
    return done


def termination_fn_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    height = next_obs[0]
    angle = next_obs[1]
    not_done = (
        jnp.isfinite(next_obs).all()
        * jnp.abs(next_obs[1:] < 100).all()
        * (height > 0.7)
        * (jnp.abs(angle) < 0.2)
    )

    done = ~not_done
    return done


def termination_fn_halfcheetahveljump(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    done = jnp.array(False)
    return done


def termination_fn_antangle(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    x = next_obs[0]
    not_done = jnp.isfinite(next_obs).all() * (x >= 0.2) * (x <= 1.0)

    done = ~not_done
    return done


def termination_fn_ant(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    x = next_obs[0]
    not_done = jnp.isfinite(next_obs).all() * (x >= 0.2) * (x <= 1.0)

    done = ~not_done
    return done


def termination_fn_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    height = next_obs[0]
    angle = next_obs[1]
    not_done = (
        jnp.logical_and(jnp.all(next_obs > -100), jnp.all(next_obs < 100))
        * (height > 0.8)
        * (height < 2.0)
        * (angle > -1.0)
        * (angle < 1.0)
    )
    done = ~not_done
    return done


def termination_fn_point2denv(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    done = jnp.array(False)
    return done


def termination_fn_point2dwallenv(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    done = jnp.array(False)
    return done


def termination_fn_pendulum(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    done = jnp.array(False)
    return done


def termination_fn_humanoid(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    z = next_obs[0]
    done = (z < 1.0) + (z > 2.0)

    return done


def termination_fn_pen(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    obj_pos = next_obs[24:27]
    done = obj_pos[2] < 0.075

    return done


def terminaltion_fn_door(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    done = jnp.array(False)

    return done


def termination_fn_relocate(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    done = jnp.array(False)

    return done


def maze2d_open_termination_fn(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    agent_location = jnp.array([obs[0], obs[1]])
    goal_location = jnp.array([2, 3])
    done = jnp.linalg.norm(agent_location - goal_location) < 0.5
    return done


def maze2d_umaze_termination_fn(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    agent_location = jnp.array([obs[0], obs[1]])
    goal_location = jnp.array([1, 1])
    done = jnp.linalg.norm(agent_location - goal_location) < 0.5
    return done


def maze2d_medium_termination_fn(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    agent_location = jnp.array([obs[0], obs[1]])
    goal_location = jnp.array([6, 6])
    done = jnp.linalg.norm(agent_location - goal_location) < 0.5
    return done


def maze2d_large_termination_fn(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    agent_location = jnp.array([obs[0], obs[1]])
    goal_location = jnp.array([7, 9])
    done = jnp.linalg.norm(agent_location - goal_location) < 0.5
    return done


def termination_fn_kitchen(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    # Implementing termination function is tricky, since it's unclear how it works in
    # the original code (see d4rl/kitchen/kitchen_envs.py)
    # Not terminating the episode works as well, it is even an argument defined in gym.
    done = jnp.array(False)
    return done


def get_termination_fn(task):
    if "halfcheetahvel" in task:
        return termination_fn_halfcheetahveljump
    elif "halfcheetah" in task:
        return termination_fn_halfcheetah
    elif "hopper" in task:
        return termination_fn_hopper
    elif "antangle" in task:
        return termination_fn_antangle
    elif "ant" in task:
        return termination_fn_ant
    elif "walker2d" in task:
        return termination_fn_walker2d
    elif "point2denv" in task:
        return termination_fn_point2denv
    elif "point2dwallenv" in task:
        return termination_fn_point2dwallenv
    elif "pendulum" in task:
        return termination_fn_pendulum
    elif "humanoid" in task:
        return termination_fn_humanoid
    elif "maze2d-open" in task:
        return maze2d_open_termination_fn
    elif "maze2d-umaze" in task:
        return maze2d_umaze_termination_fn
    elif "maze2d-medium" in task:
        return maze2d_medium_termination_fn
    elif "maze2d-large" in task:
        return maze2d_large_termination_fn
    elif "pen" in task:
        return termination_fn_pen
    elif "door" in task:
        return terminaltion_fn_door
    elif "relocate" in task:
        return termination_fn_relocate
    elif "kitchen" in task:
        return termination_fn_kitchen
    else:
        raise ValueError(f"Unknown task: {task}")
