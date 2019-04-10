import copy
import numpy as np
from . import Agent
from reaver.envs.base import Env, MultiProcEnv


def getEvents(obs):
    events = {
        'player_id':                 [obs[f][0] for f in range(obs.shape[0])],
        'minerals':                  [obs[f][1] for f in range(obs.shape[0])],
        'vespene':                   [obs[f][2] for f in range(obs.shape[0])],
        'food_used':                 [obs[f][3] for f in range(obs.shape[0])],
        'food_cap':                  [obs[f][4] for f in range(obs.shape[0])],
        'food_army':                 [obs[f][5] for f in range(obs.shape[0])],
        'food_workers':              [obs[f][6] for f in range(obs.shape[0])],
        'idle_worker_count':         [obs[f][7] for f in range(obs.shape[0])],  # negative effect on increment
        'army_count':                [obs[f][8] for f in range(obs.shape[0])],
        'warp_gate_count':           [obs[f][9] for f in range(obs.shape[0])],
        'larva_count':              [obs[f][10] for f in range(obs.shape[0])],
        'score':                    [obs[f][11] for f in range(obs.shape[0])],
        'idle_production_time':     [obs[f][12] for f in range(obs.shape[0])],  # negative effect on increment
        'idle_worker_time':         [obs[f][13] for f in range(obs.shape[0])],  # negative effect on increment
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
    event_triggers = np.zeros([len(current_events), len(current_events[0])])

    for env_no in range(event_triggers.shape[0]):
        for event_idx in range(event_triggers.shape[1]):
            if event_idx == 7 or event_idx == 12 or event_idx == 13:
                event_triggers[env_no][event_idx] = 0

                # if current_events[env_no][event_idx] == 0:
                #     event_triggers[env_no][event_idx] = 0
                # elif current_events[env_no][event_idx] > 0:
                #     event_triggers[env_no][event_idx] = -1

                # if current_events[env_no][event_idx] == previous_events[env_no][event_idx]:
                #     event_triggers[env_no][event_idx] = 0
                # elif current_events[env_no][event_idx] < previous_events[env_no][event_idx]:
                #     event_triggers[env_no][event_idx] = 1
                # elif current_events[env_no][event_idx] > previous_events[env_no][event_idx]:
                #     event_triggers[env_no][event_idx] = -1
            elif event_idx == 1 or event_idx == 2 or event_idx == 18 or event_idx == 19 :
                if current_events[env_no][event_idx] == previous_events[env_no][event_idx]:
                    event_triggers[env_no][event_idx] = 0
                elif current_events[env_no][event_idx] < previous_events[env_no][event_idx]:
                    event_triggers[env_no][event_idx] = -0.1
                elif current_events[env_no][event_idx] > previous_events[env_no][event_idx]:
                    event_triggers[env_no][event_idx] = 0.1
            else:
                if current_events[env_no][event_idx] == previous_events[env_no][event_idx]:
                    event_triggers[env_no][event_idx] = 0
                elif current_events[env_no][event_idx] < previous_events[env_no][event_idx]:
                    event_triggers[env_no][event_idx] = -1
                elif current_events[env_no][event_idx] > previous_events[env_no][event_idx]:
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

    def run(self,  env: Env, expt, event_buffer=None,  n_steps=1000000):
        env = self.wrap_env(env)
        env.start()
        try:
            self._run(env, expt, event_buffer, n_steps)
        except KeyboardInterrupt:
            env.stop()
            self.on_finish()

    def _run(self, env, expt, event_buffer, n_steps):
        self.on_start()
        obs, *_ = env.reset()
        obs = [o.copy() for o in obs]
        envs = len(obs[0])

        episode_rewards = np.zeros([envs, 1])
        final_rewards = np.zeros([envs, 1])

        episode_intrinsic_rewards = np.zeros([envs, 1])
        final_intrinsic_rewards = np.zeros([envs, 1])

        episode_events = np.zeros([envs, 24])
        final_events = np.zeros([envs, 24])


        event_triggers = None
        previous_events = None

        with open(expt.event_log_path + "/event_log.txt", 'w') as f:
            f.write("")
        f.close()

        for step in range(self.start_step, self.start_step + n_steps):

            # choose action and predict value
            action, value = self.get_action_and_value(obs)

            # take action and observe effects
            self.next_obs, reward, done, current_events = env.step(action)

            if done[0]:
                previous_events = None
                event_triggers = None

            # event perception
            # current_events_map = getEvents(current_events)

            intrinsic_reward = []
            rew = np.zeros(len(reward))
            rew = np.expand_dims(np.stack(reward), 1)

            if previous_events is not None:
                event_triggers = getTriggeredEvents(previous_events, current_events)

                for env_id in range(len(event_triggers)):
                    newReward = 0
                    for event in event_triggers[env_id]:
                            newReward += event
                    rew[env_id] = newReward
            else:
                event_triggers = np.zeros([len(current_events), len(current_events[0])])

            # determine the importance of triggered events based on their rarity
            # if event_triggers is not None and event_buffer is not None:
                # with open(expt.event_log_path + "/event_log.txt", 'a+') as f:
                #     for item in event_triggers:
                #         f.write("[ %s ]" % item)
                #     f.write("\n")
                # f.close()

            for e in event_triggers:
                intrinsic_reward.append(event_buffer.intrinsic_reward(e))

            intrinsic_reward = np.expand_dims(np.stack(intrinsic_reward), 1)
            episode_rewards += reward
            episode_intrinsic_rewards += intrinsic_reward
            episode_events += event_triggers



            # reward =  torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            # intrinsic_reward = torch.from_numpy(np.expand_dims(np.stack(intrinsic_reward), 1)).float()
            # events = torch.from_numpy(events).float()
            #

            #
            # # Event stats
            # event_rewards = []
            # for ei in range(0, args.num_events):
            #     ev = np.zeros(args.num_events)
            #     ev[ei] = 1
            #     er = event_buffer.intrinsic_reward(ev)
            #     event_rewards.append(er)
            #
            # event_episode_rewards.append(event_rewards)
            #
            # # If done then clean the history of observations.
            # masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            # final_rewards *= masks
            # final_intrinsic_rewards *= masks
            # final_events *= masks
            #
            # final_rewards += (1 - masks) * episode_rewards
            # final_intrinsic_rewards += (1 - masks) * episode_intrinsic_rewards
            # final_events += (1 - masks) * episode_events

            if event_triggers is not None:
                for i in range(len(event_triggers)):
                    if done[i]:
                        event_buffer.record_events(np.copy(episode_events[i]), frame=step)

            self.on_step(step, obs, action, reward, done, value, event_triggers)

            # intrinsic_reward = []
            # for e in current_events:
            #     intrinsic_reward.append(event_buffer.intrinsic_reward(e))

            previous_events = np.copy(current_events)
            # if not done[env_id]:
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
