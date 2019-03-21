import copy
import numpy

from reaver.envs.sc2 import processEvents
from . import Agent
from reaver.envs.base import Env, MultiProcEnv
import time


def getEvents(obs):
    events = {
        # 'player_id':                [obs[f][0] for f in range(obs.shape[0])],
        'minerals':                  [obs[f][1] for f in range(obs.shape[0])],
        'vespene':                   [obs[f][2] for f in range(obs.shape[0])],
        'food_used':                 [obs[f][3] for f in range(obs.shape[0])],
        'food_cap':                  [obs[f][4] for f in range(obs.shape[0])],
        'food_army':                 [obs[f][5] for f in range(obs.shape[0])],
        'food_workers':              [obs[f][6] for f in range(obs.shape[0])],
        'idle_worker_count':         [obs[f][7] for f in range(obs.shape[0])],
        'army_count':                [obs[f][8] for f in range(obs.shape[0])],
        'warp_gate_count':           [obs[f][9] for f in range(obs.shape[0])],
        'larva_count':              [obs[f][10] for f in range(obs.shape[0])],
        'score':                    [obs[f][11] for f in range(obs.shape[0])],
        'idle_production_time':     [obs[f][12] for f in range(obs.shape[0])],
        'idle_worker_time':         [obs[f][13] for f in range(obs.shape[0])],
        'total_value_units':        [obs[f][14] for f in range(obs.shape[0])],
        'total_value_structures':   [obs[f][15] for f in range(obs.shape[0])],
        'killed_value_units':       [obs[f][16] for f in range(obs.shape[0])],
        'killed_value_structures':  [obs[f][17] for f in range(obs.shape[0])],
        'collected_minerals':       [obs[f][18] for f in range(obs.shape[0])],
        'collected_vespene':        [obs[f][19] for f in range(obs.shape[0])],
        'collection_rate_minerals': [obs[f][20] for f in range(obs.shape[0])],
        'collection_rate_vespene':  [obs[f][21] for f in range(obs.shape[0])],
        'spent_minerals':           [obs[f][22] for f in range(obs.shape[0])],
        'spent_vespene':            [obs[f][23] for f in range(obs.shape[0])],
    }

    return events


def getTriggeredEvents(previous_events, current_events):
    event_triggers = numpy.copy(previous_events)

    for env_no in range(event_triggers.shape[0]):
        for event_idx in range(event_triggers.shape[1]):
            if previous_events[env_no][event_idx] == current_events[env_no][event_idx]:
                event_triggers[env_no][event_idx] = 0
            if previous_events[env_no][event_idx] > current_events[env_no][event_idx]:
                event_triggers[env_no][event_idx] = -1
            if previous_events[env_no][event_idx] < current_events[env_no][event_idx]:
                event_triggers[env_no][event_idx] = 1

        # elif previous_events[][env_no] == current_events[][env_no]:
        #     event_triggers[][env_no] = 0
        # elif previous_events[][env_no] < current_events[][env_no]:
        #     event_triggers[][env_no] = 1

    return event_triggers


class RunningAgent(Agent):
    """
    Generic abstract class, defines API for interacting with an environment
    """

    def __init__(self):
        self.next_obs = None
        self.start_step = 0
        self.visualise = False

    def run(self, env: Env, event_buffer=None, n_steps=1000000):
        env = self.wrap_env(env)
        env.start()
        try:
            self._run(env, event_buffer, n_steps)
        except KeyboardInterrupt:
            env.stop()
            self.on_finish()

    def _run(self, env, event_buffer, n_steps):
        self.on_start()
        obs, *_ = env.reset()
        obs = [o.copy() for o in obs]

        previous_events = None

        for step in range(self.start_step, self.start_step + n_steps):
            # if self.visualise:
            #     time.sleep(1 / 24)

            action, value = self.get_action_and_value(obs)

            # if event_buffer is None:
            #     self.next_obs, reward, done = env.step(action)
            #     self.on_step(step, obs, action, reward, done, value)
            # else:

            self.next_obs, reward, done, current_events = env.step(action)
            current_events_map = getEvents(current_events)
            if previous_events is not None:
                event_triggers = getTriggeredEvents(current_events, previous_events)

            self.on_step(step, obs, action, reward, done, value, current_events_map)

            # intrinsic_reward = []
            # for e in current_events:
            #     intrinsic_reward.append(event_buffer.intrinsic_reward(e))

            previous_events = numpy.copy(current_events)
            # previous_events_map = current_events_map

            obs = [o.copy() for o in self.next_obs]

        env.stop()
        self.on_finish()

    def get_action_and_value(self, obs):
        return self.get_action(obs), None

    def on_start(self):
        ...

    def on_step(self, step, obs, action, reward, done, value=None, events=None):
        ...

    def on_finish(self):
        ...

    def wrap_env(self, env: Env) -> Env:
        return env


class SyncRunningAgent(RunningAgent):
    """
    Abstract class that handles synchronous multiprocessing via MultiProcEnv helper
    Not meant to be used directly, extending classes automatically get the feature
    """

    def __init__(self, n_envs):
        RunningAgent.__init__(self)
        self.n_envs = n_envs

    def wrap_env(self, env: Env) -> Env:
        render, env.render = env.render, False
        envs = [env] + [copy.deepcopy(env) for _ in range(self.n_envs - 1)]
        env.render = render

        return MultiProcEnv(envs)
