import copy
import pickle
import numpy as np
from . import Agent
from reaver.envs.base import Env, MultiProcEnv

# curated_events = [
# all_events_ind[11],  # [ 0 ] = score
# all_events_ind[20],  # [ 1 ] = collection rate minerals
# all_events_ind[21],  # [ 2 ] = collection rate vespene
# all_events_ind[18],  # [ 3 ] = collected minerals
# all_events_ind[19],  # [ 4 ] = collected vespene
#
# all_events_ind[3],   # [ 5 ] = food used
# all_events_ind[4],   # [ 6 ] = food cap
# all_events_ind[6],   # [ 7 ] = food workers
#
# all_events_ind[12],  # [ 8 ] = idle prod time
# all_events_ind[7],   # [ 9 ] = idle workers
#
# all_events_ind[14],  # [ 10 ] = total value units
# all_events_ind[15],  # [ 11 ] = total value structures

# all_events_ind[1],        # [ 1 ] = minerals
# all_events_ind[2],        # [ 2 ] = vespene
# all_events_ind[22],       # [ 7 ] = spent minerals
# all_events_ind[23],       # [ 8 ] = spent vespene
# all_events_ind[7],        # [ 13 ] = idle workers
# all_events_ind[5],        # [ 14 ] = food army
# all_events_ind[8],        # [ 15 ] = army count
# all_events_ind[14],       # [ 16 ] = total value units
# all_events_ind[15],       # [ 17 ] = total value structures
# all_events_ind[16],       # [ 18 ] = killed value units
# all_events_ind[17]        # [ 19 ] = killed value structures
# ]


def getTriggeredEvents(event_buffer, previous_events, current_events):
    event_triggers = np.zeros([len(current_events), len(current_events[0])])

    mean = event_buffer.get_event_mean()

    for env_no in range(event_triggers.shape[0]):
        for event_idx in range(event_triggers.shape[1]):
            if event_idx in [0, 3, 4]:
                # strictly positive rewards
                # collected minerals and gas never decrease
                if current_events[env_no][event_idx] > previous_events[env_no][event_idx]:
                    event_triggers[env_no][event_idx] = previous_events[env_no][event_idx]

            elif event_idx in [1, 2]:
                # mixed increments positive NSF
                # small rewards
                event_triggers[env_no][event_idx] = current_events[env_no][event_idx] -\
                                                    previous_events[env_no][event_idx]

            elif event_idx in [5, 6, 7, 10, 11]:
                # mixed increments positive NSF
                # large negative rewards
                if current_events[env_no][event_idx] == previous_events[env_no][event_idx]:
                    event_triggers[env_no][event_idx] = previous_events[env_no][event_idx] - mean[event_idx]
                elif current_events[env_no][event_idx] < previous_events[env_no][event_idx]:
                    event_triggers[env_no][event_idx] = (-1) * previous_events[env_no][event_idx]
                elif current_events[env_no][event_idx] > previous_events[env_no][event_idx]:
                    event_triggers[env_no][event_idx] = previous_events[env_no][event_idx]

            elif event_idx in [8]:
                # negative effect on increment
                # small rewards
                # idle production time & idle workers count
                if current_events[env_no][event_idx] == previous_events[env_no][event_idx]:
                    event_triggers[env_no][event_idx] = previous_events[env_no][event_idx]

                elif current_events[env_no][event_idx] > previous_events[env_no][event_idx] >= mean[event_idx]:
                    event_triggers[env_no][event_idx] = current_events[env_no][event_idx] - \
                                                            mean[event_idx]
            elif event_idx in [9]:
                # positive increments negative NSF
                # small rewards
                event_triggers[env_no][event_idx] = previous_events[env_no][event_idx] - \
                                                    current_events[env_no][event_idx]

            # elif event_idx in [9]:
            #     # mixed increments positive NSF
            #     # small rewards
            #     event_triggers[env_no][event_idx] = current_events[env_no][event_idx] -\
            #                                         previous_events[env_no][event_idx]

            # elif event_idx in [10, 11]:
            #
            # elif event_idx == 15:
            #     # total value units fixes
            #     # it increases by 600 after each game
            #     if current_events[env_no][event_idx] == previous_events[env_no][event_idx]:
            #         event_triggers[env_no][event_idx] = 0
            #     elif current_events[env_no][event_idx] == previous_events[env_no][event_idx] + 600:
            #         event_triggers[env_no][event_idx] = 0
            #     elif current_events[env_no][event_idx] < previous_events[env_no][event_idx]:
            #         event_triggers[env_no][event_idx] = -1
            #     elif current_events[env_no][event_idx] > previous_events[env_no][event_idx]:
            #         event_triggers[env_no][event_idx] = 1
            #
            # elif event_idx == 16:
            #     # total value structure fixes
            #     # it increases by 400 after each game
            #     if current_events[env_no][event_idx] == previous_events[env_no][event_idx]:
            #         event_triggers[env_no][event_idx] = 0
            #     elif current_events[env_no][event_idx] == previous_events[env_no][event_idx] + 400:
            #         event_triggers[env_no][event_idx] = 0
            #     elif current_events[env_no][event_idx] < previous_events[env_no][event_idx]:
            #         event_triggers[env_no][event_idx] = -1
            #     elif current_events[env_no][event_idx] > previous_events[env_no][event_idx]:
            #         event_triggers[env_no][event_idx] = 1
            # else:
            #     if current_events[env_no][event_idx] == previous_events[env_no][event_idx]:
            #         event_triggers[env_no][event_idx] = 0
            #     elif current_events[env_no][event_idx] < previous_events[env_no][event_idx]:
            #         event_triggers[env_no][event_idx] = -1
            #     elif current_events[env_no][event_idx] > previous_events[env_no][event_idx]:
            #         event_triggers[env_no][event_idx] = 1

    return event_triggers


class RunningAgent(Agent):
    """
    Generic abstract class, defines API for interacting with an environment
    """

    def __init__(self):
        self.next_obs = None
        self.start_step = 0
        self.visualise = False

    def run(self, env: Env, expt, event_buffer=None, n_steps=1000000):
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

        event_number = event_buffer.get_events_number()
        envs = len(obs[0])

        episode_rewards = np.zeros([1, envs])

        episode_intrinsic_rewards = np.zeros([1, envs])

        episode_events = np.zeros([envs, event_number])
        broken_nsf = np.zeros([envs, event_number])

        # final_rewards = np.zeros([1, envs])
        # final_intrinsic_rewards = np.zeros([1, envs])
        # final_events = np.zeros([envs, event_number])

        # event_triggers = None

        previous_events = None
        previous_broken_nsf = None

        with open(expt.event_log_txt, 'w') as outfile:
            np.savetxt(outfile, np.arange(event_number).reshape(1, event_number), fmt="%10.0f", delimiter="|")
        outfile.close()

        for step in range(self.start_step, self.start_step + n_steps):
            # choose action and predict value
            action, value = self.get_action_and_value(obs)

            # take action and observe effects
            self.next_obs, reward, done, current_events = env.step(action)

            # fix broken non spatial feature for idle production time
            if previous_events is None:
                for i in range(len(done)):
                    broken_nsf[i][8] = current_events[i][8]
                    # broken_nsf[i][10] = current_events[i][10]
                    # broken_nsf[i][11] = current_events[i][11]

            elif previous_events is not None and done[0]:
                previous_broken_nsf = np.copy(broken_nsf)
                for i in range(len(done)):
                    broken_nsf[i][8] = current_events[i][8]
                    current_events[i][8] -= broken_nsf[i][8]

                    broken_nsf[i][10] = current_events[i][10]
                    current_events[i][10] -= broken_nsf[i][10]

                    broken_nsf[i][11] = current_events[i][11]
                    current_events[i][11] -= broken_nsf[i][11]
            else:
                for i in range(len(done)):
                    current_events[i][8] -= broken_nsf[i][8]
                    current_events[i][10] -= broken_nsf[i][10]
                    current_events[i][11] -= broken_nsf[i][11]

            # If done then clean the history of observations.
            if done[0]:

                # masks = 0.0
                # final_rewards *= masks
                # final_intrinsic_rewards *= masks
                # final_rewards += episode_rewards
                # final_intrinsic_rewards += episode_intrinsic_rewards

                # idle production time reset in between games
                for i in range(len(done)):
                    if previous_broken_nsf is not None:
                        if broken_nsf[i][8] > previous_broken_nsf[i][8]:
                            previous_events[i][8] = broken_nsf[i][8] - previous_broken_nsf[i][8]
                        elif broken_nsf[i][8] < previous_broken_nsf[i][8]:
                            previous_events[i][8] = broken_nsf[i][8]

                        if broken_nsf[i][10] > previous_broken_nsf[i][10]:
                            previous_events[i][10] = broken_nsf[i][10] - previous_broken_nsf[i][10]
                        elif broken_nsf[i][10] < previous_broken_nsf[i][10]:
                            previous_events[i][10] = broken_nsf[i][10]

                        if broken_nsf[i][11] > previous_broken_nsf[i][11]:
                            previous_events[i][11] = broken_nsf[i][11] - previous_broken_nsf[i][11]
                        elif broken_nsf[i][11] < previous_broken_nsf[i][11]:
                            previous_events[i][11] = broken_nsf[i][11]
                    else:
                        previous_events[i][8] = broken_nsf[i][8]
                        previous_events[i][10] = broken_nsf[i][10]
                        previous_events[i][11] = broken_nsf[i][11]

                final_events = np.zeros([envs, event_number])
                final_events = np.copy(previous_events)

                episode_rewards = np.zeros([1, envs])
                episode_intrinsic_rewards = np.zeros([1, envs])
                # episode_events = np.zeros([envs, event_number])

                previous_events = None

                for i in range(len(done)):
                    event_buffer.record_events(np.copy(final_events[i]), frame=step)

                event_str = ""
                for j in range(final_events.shape[0]):
                    for i in range(len(final_events[0])):
                        event_str += "{:2d}: {:5.0f} |".format(i, final_events[j][i])
                    event_str += "\n"
                event_str += "\n"

                with open(expt.event_log_txt, 'a') as outfile:
                    outfile.write(event_str)
                    # np.savetxt(outfile, final_events, fmt="%7.0f", delimiter="|")
                outfile.close()

                with open(expt.event_log_pkl, "wb") as f:
                    pickle.dump(event_buffer, f)
                f.close()

            intrinsic_reward = []

            if previous_events is not None:
                event_triggers = getTriggeredEvents(event_buffer, previous_events, current_events)
            else:
                event_triggers = np.zeros([len(current_events), len(current_events[0])])

            # determine the importance of triggered events based on their rarity
            for e in event_triggers:
                intrinsic_reward.append(event_buffer.intrinsic_reward(e))

            # remember reward, intrinsic rewards and events
            episode_rewards += reward
            episode_intrinsic_rewards += intrinsic_reward
            # episode_events += event_triggers
            # episode_events = np.copy(event_triggers)

            intrinsic_rew = np.array(reward, dtype=np.float64)
            for i in range(len(reward)):
                intrinsic_rew[i] = intrinsic_reward[i]

            self.on_step(step, obs, action, intrinsic_rew, game_reward=reward, done=done, value=value)

            previous_events = np.copy(current_events)
            obs = [o.copy() for o in self.next_obs]

        env.stop()
        self.on_finish()

    def get_action_and_value(self, obs):
        return self.get_action(obs), None

    def on_start(self):
        ...

    def on_step(self, step, obs, action, intrinsic_rew, game_reward, done, value=None):
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
