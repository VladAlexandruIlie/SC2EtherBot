import copy
import pickle
import numpy as np
from . import Agent
from reaver.envs.base import Env, MultiProcEnv


def make_starting_variables(expt, event_buffer, obs):
    global event_number, envs, \
        episode_intrinsic_rewards, episode_events, starting_broken_nsf, previous_broken_nsf, \
        supply_blocked_idle_production_time, previous_player_layers, previous_score_cumulative_layers

    global player_id, minerals, vespene, food_used, food_cap, food_army, food_workers, idle_worker_count, army_count, \
        warp_gate_count, larva_count, \
        score, idle_production_time, idle_worker_time, total_value_units, total_value_structures, \
        killed_value_units, killed_value_structures, collected_minerals, collected_vespene, \
        collection_rate_minerals, collection_rate_vespene, spent_minerals, spent_vespene

    global starting_values

    events_number = obs[-2].shape[1] + obs[-1].shape[1]
    event_buffer.set_event_number(events_number)
    event_number = event_buffer.get_events_number()

    envs = len(obs[0])

    # non_spatial_features_idx
    # player non spatial features and indices
    player_id = 0
    minerals = 1
    vespene = 2
    food_used = 3
    food_cap = 4
    food_army = 5
    food_workers = 6
    idle_worker_count = 7
    army_count = 8
    warp_gate_count = 9
    larva_count = 10

    # score cumulative non spatial features and indices
    score = 0
    idle_production_time = 1
    idle_worker_time = 2
    total_value_units = 3
    total_value_structures = 4
    killed_value_units = 5
    killed_value_structures = 6
    collected_minerals = 7
    collected_vespene = 8
    collection_rate_minerals = 9
    collection_rate_vespene = 10
    spent_minerals = 11
    spent_vespene = 12

    # episode_rewards = np.zeros([1, envs])
    episode_intrinsic_rewards = np.zeros([1, envs])
    episode_events = np.zeros([envs, event_number])
    starting_broken_nsf = np.zeros([envs, obs[-1].shape[1]])
    previous_broken_nsf = None
    supply_blocked_idle_production_time = 0

    # final_rewards = np.zeros([1, envs])
    # final_intrinsic_rewards = np.zeros([1, envs])
    # final_events = np.zeros([envs, event_number])
    # event_triggers = None

    with open(expt.event_log_txt, 'w') as outfile:
        np.savetxt(outfile, np.arange(event_number).reshape(1, event_number), fmt="%10.0f", delimiter="|")
    outfile.close()

    previous_player_layers = None
    previous_score_cumulative_layers = None
    starting_values = None

def extract_obs_layers(obs):
    screen_layers = np.copy(obs[0])
    minimap_layers = np.copy(obs[1])
    actions_layers = np.copy(obs[2])
    player_layers = np.copy(obs[3])
    score_cumulative_layers = np.copy(obs[4])

    return screen_layers, minimap_layers, actions_layers, player_layers, score_cumulative_layers


def remake_observations(screen_layers, minimap_layers, actions_layers, player_layers, score_cumulative_layers):
    obs_improved = [screen_layers, minimap_layers, actions_layers, player_layers, score_cumulative_layers]
    return obs_improved


def calculate_intrinsic_reward(event_buffer, event_triggers, reward):
    global episode_intrinsic_rewards
    intrinsic_reward = []
    for e in event_triggers:
        intrinsic_reward.append(event_buffer.intrinsic_reward(e))

    episode_intrinsic_rewards += intrinsic_reward

    intrinsic_rew = np.array(reward, dtype=np.float64)
    for i in range(len(reward)):
        intrinsic_rew[i] = intrinsic_reward[i]

    return intrinsic_rew


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

        # starting variables declaration
        make_starting_variables(expt, event_buffer, obs)

        # main running loop
        for step in range(self.start_step, self.start_step + n_steps):
            global previous_player_layers, previous_score_cumulative_layers, starting_values

            # choose action and predict value
            action, value = self.get_action_and_value(obs)

            # take action and observe effects
            self.next_obs, reward, done = env.step(action)

            # breakdown the obs into layers
            screen_layers, minimap_layers, actions_layers, \
                player_layers, score_cumulative_layers = extract_obs_layers(self.next_obs)

            score_cumulative_layers = remake_score_cumulative(score_cumulative_layers, previous_score_cumulative_layers,
                                                              player_layers, previous_player_layers, done)

            self.next_obs = remake_observations(screen_layers, minimap_layers, actions_layers,
                                                player_layers, score_cumulative_layers)

            # find event triggers based on non-spatial features
            previous_events = np.hstack((previous_player_layers, previous_score_cumulative_layers))
            current_events = np.hstack((player_layers, score_cumulative_layers))

            if previous_player_layers is None and previous_score_cumulative_layers is None:
                starting_values = np.copy(current_events)
            event_triggers = getTriggeredEvents(done, previous_events, current_events, starting_values)

            # calculate intrinsic reward from event_triggers and event_buffer
            intrinsic_reward = calculate_intrinsic_reward(event_buffer, event_triggers, reward)

            # remember reward, intrinsic rewards and events
            if previous_player_layers is not None and previous_score_cumulative_layers is not None:
                save_episode_events(previous_events, current_events, event_triggers, starting_values, done)

            if done[0]:
                record_final_events(step, expt, event_buffer)

            self.on_step(step, obs, action, intrinsic_reward, game_reward=reward, done=done, value=value)

            if done[0]:
                previous_player_layers = None
                previous_score_cumulative_layers = None
            else:
                previous_player_layers = np.copy(player_layers)
                previous_score_cumulative_layers = np.copy(score_cumulative_layers)

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


def remake_score_cumulative(current_score_cumulative_layers, previous_score_cumulative_layers,
                            current_player_layers, previous_player_layers,
                            done):
    global food_cap, food_army, food_workers, score, idle_production_time, idle_worker_time, total_value_units, \
        total_value_structures, killed_value_units, killed_value_structures, collected_minerals, collected_vespene, \
        collection_rate_minerals, collection_rate_vespene, spent_minerals, spent_vespene

    global supply_blocked_idle_production_time, previous_broken_nsf

    remade_cumulative_layer = np.copy(current_score_cumulative_layers)

    # increase idle production time if player is supply blocked
    if previous_score_cumulative_layers is not None:
        for i in range(len(done)):
            if current_player_layers[i][food_cap] <= current_player_layers[i][food_workers] + \
                                                     current_player_layers[i][food_army] and \
                    remade_cumulative_layer[i][idle_production_time] == \
                                    previous_score_cumulative_layers[i][idle_production_time] - \
                                    supply_blocked_idle_production_time and \
                    current_player_layers[i][food_cap] < 200:
                supply_blocked_idle_production_time += 1
                remade_cumulative_layer[i][idle_production_time] += supply_blocked_idle_production_time

    # if it's the first step in the env than remember what it started with
    if previous_score_cumulative_layers is None:
        for i in range(len(done)):
            if previous_broken_nsf is not None:
                remade_cumulative_layer[i][idle_production_time] -= starting_broken_nsf[i][idle_production_time]
                remade_cumulative_layer[i][idle_worker_time] -= starting_broken_nsf[i][idle_worker_time]
                remade_cumulative_layer[i][total_value_units] -= starting_broken_nsf[i][total_value_units]
                remade_cumulative_layer[i][total_value_structures] -= starting_broken_nsf[i][total_value_structures]
                remade_cumulative_layer[i][spent_minerals] -= starting_broken_nsf[i][spent_minerals]
                remade_cumulative_layer[i][spent_vespene] -= starting_broken_nsf[i][spent_vespene]

            else:
                starting_broken_nsf[i][idle_production_time] = remade_cumulative_layer[i][idle_production_time] - 1
                starting_broken_nsf[i][idle_worker_time] = remade_cumulative_layer[i][idle_worker_time] - 12
                starting_broken_nsf[i][total_value_units] = remade_cumulative_layer[i][total_value_units]
                starting_broken_nsf[i][total_value_structures] = remade_cumulative_layer[i][total_value_structures]
                starting_broken_nsf[i][spent_minerals] = remade_cumulative_layer[i][spent_minerals]
                starting_broken_nsf[i][spent_vespene] = remade_cumulative_layer[i][spent_vespene]

    elif previous_score_cumulative_layers is not None and done[0]:
        previous_broken_nsf = np.copy(starting_broken_nsf)
        supply_blocked_idle_production_time = 0

        for i in range(len(remade_cumulative_layer)):
            if remade_cumulative_layer[i][idle_production_time] >= previous_broken_nsf[i][idle_production_time]:
                remade_cumulative_layer[i][idle_production_time] -= previous_broken_nsf[i][idle_production_time]

            if remade_cumulative_layer[i][idle_worker_time] >= previous_broken_nsf[i][idle_worker_time]:
                remade_cumulative_layer[i][idle_worker_time] -= previous_broken_nsf[i][idle_worker_time]

            if remade_cumulative_layer[i][total_value_units] >= previous_broken_nsf[i][total_value_units]:
                remade_cumulative_layer[i][total_value_units] -= previous_broken_nsf[i][total_value_units]

            if remade_cumulative_layer[i][total_value_structures] >= previous_broken_nsf[i][total_value_structures]:
                remade_cumulative_layer[i][total_value_structures] -= previous_broken_nsf[i][total_value_structures]

            if remade_cumulative_layer[i][spent_minerals] >= previous_broken_nsf[i][spent_minerals]:
                remade_cumulative_layer[i][spent_minerals] -= previous_broken_nsf[i][spent_minerals]

            if remade_cumulative_layer[i][spent_vespene] >= previous_broken_nsf[i][spent_vespene]:
                remade_cumulative_layer[i][spent_vespene] -= previous_broken_nsf[i][spent_vespene]

            starting_broken_nsf[i][idle_production_time] = remade_cumulative_layer[i][idle_production_time]
            starting_broken_nsf[i][idle_worker_time] = remade_cumulative_layer[i][idle_worker_time]
            starting_broken_nsf[i][total_value_units] = remade_cumulative_layer[i][total_value_units]
            starting_broken_nsf[i][total_value_structures] = remade_cumulative_layer[i][total_value_structures]
            starting_broken_nsf[i][spent_minerals] = remade_cumulative_layer[i][spent_minerals]
            starting_broken_nsf[i][spent_vespene] = remade_cumulative_layer[i][spent_vespene]

    elif previous_broken_nsf is not None and not done[0]:
        for i in range(len(done)):
            if remade_cumulative_layer[i][idle_production_time] >= starting_broken_nsf[i][idle_production_time]:
                remade_cumulative_layer[i][idle_production_time] -= starting_broken_nsf[i][idle_production_time]

            remade_cumulative_layer[i][idle_production_time] += supply_blocked_idle_production_time

            if remade_cumulative_layer[i][idle_worker_time] >= starting_broken_nsf[i][idle_worker_time]:
                remade_cumulative_layer[i][idle_worker_time] -= starting_broken_nsf[i][idle_worker_time]

            if remade_cumulative_layer[i][total_value_units] >= starting_broken_nsf[i][total_value_units]:
                remade_cumulative_layer[i][total_value_units] -= starting_broken_nsf[i][total_value_units]

            if remade_cumulative_layer[i][total_value_structures] >= starting_broken_nsf[i][total_value_structures]:
                remade_cumulative_layer[i][total_value_structures] -= starting_broken_nsf[i][total_value_structures]

            if remade_cumulative_layer[i][spent_minerals] >= starting_broken_nsf[i][spent_minerals]:
                remade_cumulative_layer[i][spent_minerals] -= starting_broken_nsf[i][spent_minerals]

            if remade_cumulative_layer[i][spent_vespene] >= starting_broken_nsf[i][spent_vespene]:
                remade_cumulative_layer[i][spent_vespene] -= starting_broken_nsf[i][spent_vespene]

    return remade_cumulative_layer


def processEvents(player_layers, score_cumulative_layers):
    all_events_ind = np.hstack((player_layers, score_cumulative_layers))

    curated_events = []

    for env_number in range(player_layers.shape[0]):
        curated_events[env_number] = [
            all_events_ind[env_number][11],  # [ 0 ] = score
            all_events_ind[env_number][20],  # [ 1 ] = collection rate minerals
            all_events_ind[env_number][21],  # [ 2 ] = collection rate vespene
            all_events_ind[env_number][18],  # [ 3 ] = collected minerals
            all_events_ind[env_number][19],  # [ 4 ] = collected vespene

            all_events_ind[env_number][3],  # [ 5 ] = food used
            all_events_ind[env_number][4],  # [ 6 ] = food cap
            all_events_ind[env_number][6],  # [ 7 ] = food workers
            # all_events_ind[5],              # [ 8 ] = food army

            all_events_ind[env_number][12],  # [ 9 ] = idle prod time
            all_events_ind[env_number][7],  # [ 10 ] = idle workers

            all_events_ind[env_number][14],  # [ 11 ] = total value units
            all_events_ind[env_number][15]  # [ 12 ] = total value structures

            # all_events_ind[1],        # [ 1 ] = minerals
            # all_events_ind[2],        # [ 2 ] = vespene
            # all_events_ind[22],       # [ 7 ] = spent minerals
            # all_events_ind[23],       # [ 8 ] = spent vespene
            # all_events_ind[5],        # [ 14 ] = food army
            # all_events_ind[8],        # [ 15 ] = army count

            # all_events_ind[16],       # [ 18 ] = killed value units
            # all_events_ind[17]        # [ 19 ] = killed value structures
        ]
    return curated_events


def getTriggeredEvents(done, previous_events, current_events, starting_values):
    # non_spatial_features_idx
    # player non spatial features and indices
    player_id = 0
    minerals = 1
    vespene = 2
    food_used = 3
    food_cap = 4
    food_army = 5
    food_workers = 6
    idle_worker_count = 7
    army_count = 8
    warp_gate_count = 9
    larva_count = 10
    # score cumulative non spatial features and indices
    score = 11
    idle_production_time = 12
    idle_worker_time = 13
    total_value_units = 14
    total_value_structures = 15
    killed_value_units = 16
    killed_value_structures = 17
    collected_minerals = 18
    collected_vespene = 19
    collection_rate_minerals = 20
    collection_rate_vespene = 21
    spent_minerals = 22
    spent_vespene = 23

    event_triggers = np.zeros([len(current_events), len(current_events[0])])

    if previous_player_layers is None or previous_score_cumulative_layers is None or done[0]:
        return event_triggers

    for env_no in range(event_triggers.shape[0]):
        for event_idx in range(event_triggers.shape[1]):

            if event_idx in [score, collected_minerals, collected_vespene,
                             collection_rate_minerals, collection_rate_vespene,
                             ]:
                # large strictly positive rewards
                # to maximise
                if current_events[env_no][event_idx] > previous_events[env_no][event_idx]:
                    event_triggers[env_no][event_idx] = current_events[env_no][event_idx] - \
                                                        starting_values[env_no][event_idx]

            elif event_idx in [minerals, vespene,
                               spent_minerals, spent_vespene,
                               killed_value_units, killed_value_structures]:
                # small rewards
                event_triggers[env_no][event_idx] = current_events[env_no][event_idx] - \
                                                        previous_events[env_no][event_idx]

            elif event_idx in [food_used, food_cap, food_workers, food_army, army_count,
                               total_value_units, total_value_structures,
                               warp_gate_count, larva_count,
                               ]:
                # large positive rewards on gain
                # small negative rewards on loss
                if current_events[env_no][event_idx] > previous_events[env_no][event_idx]:
                    event_triggers[env_no][event_idx] = current_events[env_no][event_idx] - \
                                                        starting_values[env_no][event_idx]
                else:
                    event_triggers[env_no][event_idx] = current_events[env_no][event_idx] - \
                                                        previous_events[env_no][event_idx]

            elif event_idx in [idle_worker_count]:
                # idle workers count
                # positive increments negative NSF
                # small rewards
                # skip for now because of active worker time (from idle worker time)
                event_triggers[env_no][event_idx] = 0

            elif event_idx in [idle_production_time]:
                # negative effect on increment
                # small rewards
                # idle production time &
                if current_events[env_no][event_idx] == previous_events[env_no][event_idx] and \
                    current_events[env_no][food_cap] > current_events[env_no][food_workers] + \
                                                       current_events[env_no][food_army]:
                    event_triggers[env_no][event_idx] = 1

            elif event_idx in [idle_worker_time]:
                # negative effect on increment
                # small rewards
                # idle worker time
                if current_events[env_no][event_idx] == previous_events[env_no][event_idx]:
                    event_triggers[env_no][event_idx] = 1

    return event_triggers


def save_episode_events(previous_events, current_events, event_triggers, staring_events, done):
    global episode_events

    # non_spatial_features_idx
    # player non spatial features and indices
    player_id = 0
    minerals = 1
    vespene = 2
    food_used = 3
    food_cap = 4
    food_army = 5
    food_workers = 6
    idle_worker_count = 7
    army_count = 8
    warp_gate_count = 9
    larva_count = 10
    # score cumulative non spatial features and indices
    score = 11
    idle_production_time = 12
    idle_worker_time = 13
    total_value_units = 14
    total_value_structures = 15
    killed_value_units = 16
    killed_value_structures = 17
    collected_minerals = 18
    collected_vespene = 19
    collection_rate_minerals = 20
    collection_rate_vespene = 21
    spent_minerals = 22
    spent_vespene = 23

    episode_events += event_triggers

    if previous_events is not None and not done[0]:
        for env_no in range(len(previous_events)):
            episode_events[env_no][score] = current_events[env_no][score]
            episode_events[env_no][food_cap] = current_events[env_no][food_cap] - staring_events[env_no][food_cap]
            episode_events[env_no][food_army] = current_events[env_no][food_army] - staring_events[env_no][food_army]
            episode_events[env_no][food_workers] = current_events[env_no][food_workers] - staring_events[env_no][food_workers]
            episode_events[env_no][food_used] = current_events[env_no][food_used] - staring_events[env_no][food_used]

            episode_events[env_no][total_value_units] = current_events[env_no][total_value_units] - \
                                                                staring_events[env_no][total_value_units]
            episode_events[env_no][total_value_structures] = current_events[env_no][total_value_structures] - \
                                                                staring_events[env_no][total_value_structures]

            episode_events[env_no][collected_minerals] = current_events[env_no][collected_minerals]
            episode_events[env_no][collected_vespene] = current_events[env_no][collected_vespene]
            episode_events[env_no][collection_rate_minerals] = current_events[env_no][collection_rate_minerals]
            episode_events[env_no][collection_rate_vespene] = current_events[env_no][collection_rate_vespene]
            episode_events[env_no][idle_worker_count] = current_events[env_no][idle_worker_count]

            # episode_events[env_no][0:5] = current_events[env_no][0:5]
            # episode_events[env_no][9] = current_events[env_no][9]


def record_final_events(step, expt, event_buffer):
    global episode_events, episode_intrinsic_rewards, previous_broken_nsf, starting_broken_nsf

    final_events = np.zeros([envs, event_number])
    final_events += episode_events

    episode_intrinsic_rewards = np.zeros([1, envs])
    episode_events = np.zeros([envs, event_number])

    for i in range(len(final_events)):
        event_buffer.record_events(np.copy(final_events[i]), frame=step)

    event_str = ""
    for j in range(final_events.shape[0]):
        for i in range(len(final_events[0])):
            event_str += "{:2d}: {:5.0f} |".format(i, final_events[j][i])
        event_str += "\n"
    event_str += "\n"

    with open(expt.event_log_txt, 'a') as outfile:
        outfile.write(event_str)
    outfile.close()

    with open(expt.event_log_pkl, "wb") as f:
        pickle.dump(event_buffer, f)
    f.close()

# curated_events = [
# all_events_ind[11],   # [ 0 ] = score
# all_events_ind[20],   # [ 1 ] = collection rate minerals
# all_events_ind[21],   # [ 2 ] = collection rate vespene
# all_events_ind[18],   # [ 3 ] = collected minerals
# all_events_ind[19],   # [ 4 ] = collected vespene
#
# all_events_ind[3],    # [ 5 ] = food used
# all_events_ind[4],    # [ 6 ] = food cap
# all_events_ind[6],    # [ 7 ] = food workers
# all_events_ind[5],    # [ 8 ] = food army
#
# all_events_ind[12],  #  [ 9 ] = idle prod time
# all_events_ind[7],   # [ 10 ] = idle workers
#
# all_events_ind[14],  # [ 11 ] = total value units
# all_events_ind[15],  # [ 12 ] = total value structures
# ]
