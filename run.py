import os
import pickle

import gin
import tensorflow as tf
from absl import app, flags
import numpy
import reaver as rvr
from roe_utils.event_buffer import EventBufferSQLProxy, EventBuffer

numpy.warnings.filterwarnings('ignore')

flags.DEFINE_string('env', None, 'Either Gym env id or PySC2 map name to run agent in.')
flags.DEFINE_string('agent', 'a2c', 'Name of the agent. Must be one of (a2c, ppo).')

flags.DEFINE_bool('render', False, 'Whether to render first(!) env.')
flags.DEFINE_string('gpu', "0", 'GPU(s) id(s) to use. If not set TensorFlow will use CPU.')

flags.DEFINE_integer('n_envs', 1, 'Number of environments to execute in parallel.')
flags.DEFINE_integer('n_updates', 1000000, 'Number of train updates (1 update has batch_sz * traj_len samples).')

flags.DEFINE_integer('ckpt_freq', 250, 'Number of train updates per one checkpoint save.')
flags.DEFINE_integer('log_freq', 20, 'Number of train updates per one console log.')
flags.DEFINE_integer('log_eps_avg', 100, 'Number of episodes to average for performance stats.')
flags.DEFINE_integer('max_ep_len', None, 'Max number of steps an agent can take in an episode.')

flags.DEFINE_string('results_dir', 'results', 'Directory for model weights, train logs, etc.')
flags.DEFINE_string('experiment', None, 'Name of the experiment. Datetime by default.')

flags.DEFINE_multi_string('gin_files', [], 'List of path(s) to gin config(s).')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin bindings to override config values.')

flags.DEFINE_bool('restore', False,
                  'Restore & continue previously executed experiment. '
                  'If experiment not specified then last modified is used.')

flags.DEFINE_bool('test', False,
                  'Run an agent in test mode: restore flag is set to true and number of envs set to 1'
                  'Loss is calculated, but gradients are not applied.'
                  'Checkpoints, summaries, log files are not updated, but console logger is enabled.')

flags.DEFINE_bool('roe', True,
                  'Trains using Rairty of Events (default: False)')
flags.DEFINE_integer('num-events', 4,
                     'number of events to record (default: 4)')
flags.DEFINE_integer('capacity', 100,
                     'Size of the event buffer (default: 100)')
flags.DEFINE_bool('qd', False,
                  'RoE QD (default: False)')

flags.DEFINE_alias('e', 'env')
flags.DEFINE_alias('a', 'agent')
flags.DEFINE_alias('p', 'n_envs')
flags.DEFINE_alias('u', 'n_updates')
flags.DEFINE_alias('lf', 'log_freq')
flags.DEFINE_alias('cf', 'ckpt_freq')
flags.DEFINE_alias('la', 'log_eps_avg')
flags.DEFINE_alias('n', 'experiment')
flags.DEFINE_alias('g', 'gin_bindings')


def main(argv):
    # flags and CPU vs. GPU
    args = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # map aliases
    if args.env in rvr.utils.config.SC2_MINIGAMES_ALIASES:
        args.env = rvr.utils.config.SC2_MINIGAMES_ALIASES[args.env]

    # test mode
    if args.test:
        args.n_envs = 1
        args.log_freq = 1
        args.restore = True

    # create experiments directories
    if args.roe is True:
        expt = rvr.utils.Experiment(args.results_dir, args.env, args.agent + "+roe", args.experiment, args.restore)
    else:
        expt = rvr.utils.Experiment(args.results_dir, args.env, args.agent, args.experiment, args.restore)

    # set-up gin
    gin_files = rvr.utils.find_configs(args.env, os.path.dirname(os.path.abspath(__file__)) + "/reaver")

    # restore point
    if args.restore:
        gin_files += [expt.config_path]
    gin_files += args.gin_files

    # changes for CPU
    if not args.gpu:
        args.gin_bindings.append("build_cnn_nature.data_format = 'channels_last'")
        args.gin_bindings.append("build_fully_conv.data_format = 'channels_last'")
    gin.parse_config_files_and_bindings(gin_files, args.gin_bindings)
    args.n_envs = min(args.n_envs, gin.query_parameter('ACAgent.batch_sz'))

    # start tensorflow
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess_mgr = rvr.utils.tensorflow.SessionManager(sess, expt.path, args.ckpt_freq, training_enabled=not args.test)

    # make environments
    env_cls = rvr.envs.GymEnv if '-v' in args.env else rvr.envs.SC2Env
    env = env_cls(args.env, args.render, max_ep_len=args.max_ep_len)

    # make an A2C Agent & envs
    agent = rvr.agents.registry[args.agent](env.obs_spec(), env.act_spec(), sess_mgr=sess_mgr, n_envs=args.n_envs)
    agent.logger = rvr.utils.StreamLogger(args.n_envs, args.log_freq, args.log_eps_avg, sess_mgr, expt.log_path)

    # first time save
    if sess_mgr.training_enabled:
        expt.save_gin_config()
        expt.save_model_summary(agent.model)

    # ROE temp variables
    episode_rewards = numpy.zeros([args.n_envs, 1])
    final_rewards = numpy.zeros([args.n_envs, 1])
    episode_intrinsic_rewards = numpy.zeros([args.n_envs, 1])
    final_intrinsic_rewards = numpy.zeros([args.n_envs, 1])
    episode_events = numpy.zeros([args.n_envs, args.num_events])
    final_events = numpy.zeros([args.n_envs, args.num_events])

    # Create event buffer
    if args.qd:
        event_buffer = EventBufferSQLProxy(args.num_events, args.capacity, args.exp_id, args.agent_id)
    elif not args.restore:
        event_buffer = EventBuffer(args.num_events, args.capacity)
    else:
        event_buffer = pickle.load(open(expt.event_log_path + args.env + "_event_buffer_temp.p", "rb"))

    agent.run(env, event_buffer, args.n_updates * agent.traj_len * agent.batch_sz // args.n_envs)


if __name__ == '__main__':
    flags.mark_flag_as_required('env')
    app.run(main)
