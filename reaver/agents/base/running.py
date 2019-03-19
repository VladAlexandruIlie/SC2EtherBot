import copy
from . import Agent
from reaver.envs.base import Env, MultiProcEnv
import time


def processEvents(obs):
    # _events_ind = []
    # for feature_map_idx in range(3, len(obs)):
    #     for i in obs[feature_map_idx]:
    #         _events_ind.append(i)

    events = {
        'player_id':                [obs[f][0] for f in range(obs.shape[0])],
        'minerals':                 [obs[f][1] for f in range(obs.shape[0])],
        'vespene':                  [obs[f][2] for f in range(obs.shape[0])],
        'food_used':                [obs[f][3] for f in range(obs.shape[0])],
        'food_cap':                 [obs[f][4] for f in range(obs.shape[0])],
        'food_army':                [obs[f][5] for f in range(obs.shape[0])],
        'food_workers':             [obs[f][6] for f in range(obs.shape[0])],
        'idle_worker_count':        [obs[f][7] for f in range(obs.shape[0])],
        'army_count':               [obs[f][8] for f in range(obs.shape[0])],
        'warp_gate_count':          [obs[f][9] for f in range(obs.shape[0])],
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

        for step in range(self.start_step, self.start_step + n_steps):
            # if self.visualise:
            #     time.sleep(1 / 24)

            action, value = self.get_action_and_value(obs)

            # if event_buffer is None:
            #     self.next_obs, reward, done = env.step(action)
            #     self.on_step(step, obs, action, reward, done, value)
            # else:

            self.next_obs, reward, done, event_indicators = env.step(action)
            event_map = processEvents(event_indicators)
            self.on_step(step, obs, action, reward, done, value, event_indicators)

            intrinsic_reward = []
            for e in event_indicators:
                intrinsic_reward.append(event_buffer.intrinsic_reward(e))

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
