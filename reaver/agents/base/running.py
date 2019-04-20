import copy
import pickle
import numpy as np
from . import Agent
from reaver.envs.base import Env, MultiProcEnv


# curated_events = [all_events_ind[11],  #  [ 0 ] = score
#                   all_events_ind[1],   #  [ 1 ] = minerals
#                   all_events_ind[2],   #  [ 2 ] = vespene
#                   all_events_ind[18],  #  [ 3 ] = collected minerals
#                   all_events_ind[19],  #  [ 4 ] = collected vespene
#                   all_events_ind[20],  #  [ 5 ] = collection rate minerals
#                   all_events_ind[21],  #  [ 6 ] = collection rate vespene
#                   all_events_ind[22],  #  [ 7 ] = spent minerals
#                   all_events_ind[23],  #  [ 8 ] = spent vespene
#                   all_events_ind[3],   #  [ 9 ] = food used
#                   all_events_ind[4],   # [ 10 ] = food cap
#                   all_events_ind[5],   # [ 11 ] = food army
#                   all_events_ind[6],   # [ 12 ] = food workers
#                   all_events_ind[7],   # [ 13 ] = idle workers
#                   all_events_ind[8],   # [ 14 ] = army count
#                   all_events_ind[14],  # [ 15 ] = total value units
#                   all_events_ind[15],  # [ 16 ] = total value structures
#                   all_events_ind[16],  # [ 17 ] = killed value units
#                   all_events_ind[17]   # [ 18 ] = killed value structures
#                   ]


def getTriggeredEvents(previous_events, current_events):
    event_triggers = np.zeros([len(current_events), len(current_events[0])])

    for env_no in range(event_triggers.shape[0]):
        for event_idx in range(event_triggers.shape[1]):
            if current_events[env_no][event_idx] > previous_events[env_no][event_idx]:
                event_triggers[env_no][event_idx] = current_events[env_no][event_idx]

        # if event_idx == 0:
        #     # score indicator fix
        #     if current_events[env_no][event_idx] == previous_events[env_no][event_idx]:
        #         event_triggers[env_no][event_idx] = 0
        #     elif current_events[env_no][event_idx] > previous_events[env_no][event_idx]:
        #         event_triggers[env_no][event_idx] = 1
        # elif event_idx == 13:
        #     # idle workers
        #     #  reward =  -( idle_workers_no / workers_no )
        #     if current_events[env_no][event_idx] > 0:
        #         event_triggers[env_no][event_idx] = - current_events[env_no][event_idx] \
        #                                             / current_events[env_no][12]
        #     elif current_events[env_no][event_idx] == 0:
        #         event_triggers[env_no][event_idx] = 1
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
        envs = len(obs[0])

        event_number = 3

        # episode_rewards = np.zeros([1, envs])
        # episode_intrinsic_rewards = np.zeros([1, envs])
        # episode_events = np.zeros([envs, event_number
        # final_rewards = np.zeros([1, envs])
        # final_intrinsic_rewards = np.zeros([1, envs])
        # final_events = np.zeros([envs, event_number])

        # event_triggers = None
        previous_events = None

        with open(expt.event_log_txt, 'w') as outfile:
            np.savetxt(outfile, np.arange(event_number).reshape(1, event_number), fmt="%10.0f", delimiter="|")
        outfile.close()

        for step in range(self.start_step, self.start_step + n_steps):
            # choose action and predict value
            action, value = self.get_action_and_value(obs)

            # take action and observe effects
            self.next_obs, reward, done, current_events = env.step(action)

            # If done then clean the history of observations.
            if done[0]:

                # masks = 0.0

                # final_rewards *= masks
                # final_intrinsic_rewards *= masks
                # final_events = np.zeros([envs, event_number])

                # final_rewards += episode_rewards
                # final_intrinsic_rewards += episode_intrinsic_rewards
                final_events = np.copy(previous_events)

                # episode_rewards *= masks
                # episode_intrinsic_rewards *= masks
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
                event_triggers = getTriggeredEvents(previous_events, current_events)
            else:
                event_triggers = np.zeros([len(current_events), len(current_events[0])])

            # determine the importance of triggered events based on their rarity
            for e in event_triggers:
                intrinsic_reward.append(event_buffer.intrinsic_reward(e))

            # remember reward, intrinsic rewards and events
            # episode_rewards += reward
            # episode_intrinsic_rewards += intrinsic_reward
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
