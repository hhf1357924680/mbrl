#! pip install git+https://github.com/deepmind/pycolab.git
#! pip install git+https://github.com/openai/gym.git

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import sys
import gym

from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites as prefab_sprites


class PlayerSprite(prefab_sprites.MazeWalker):  # 定义player，从起始位置到达目标位置（0，8）
    """A `Sprite` for our player.

    This `Sprite` ties actions to going in the four cardinal directions. If we
    reach the goal state (0,8), the agent receives a
    reward of 1 and the epsiode terminates.
    """

    def __init__(self, corner, position, character):
        """Inform superclass that we can't walk through walls."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable="#", confined_to_board=True
        )

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del layers, backdrop, things  # Unused.

        # Apply motion commands. action(0,1,2,3)->(上，下，左，右)
        if actions == 0:  # walk upward?
            self._north(board, the_plot)
        elif actions == 1:  # walk downward?
            self._south(board, the_plot)
        elif actions == 2:  # walk leftward?
            self._west(board, the_plot)
        elif actions == 3:  # walk rightward?
            self._east(board, the_plot)

        #   # See if we've found the mystery spot.#是否到达目标点（0，8）
        if self.position == (0, 8):
            the_plot.add_reward(1)
            the_plot.terminate_episode()
        else:
            the_plot.add_reward(0)


# The game art for the case where
# the path on the left is open. #迷宫地图，左边开口
LEFT_OPEN_GAME_ART = [
    "        G",
    "         ",
    "         ",
    " ########",
    "         ",
    "   P     ",
]


# The game art for the case where
# the path on the right is open. #右可以走
RIGHT_OPEN_GAME_ART = [
    "        G",
    "         ",
    "         ",
    "######## ",
    "         ",
    "   P     ",
]


# The game art for the case where the path
# on both the right and left are open. #左右都可以走
BOTH_OPEN_GAME_ART = [
    "        G",
    "         ",
    "         ",
    " ####### ",
    "         ",
    "   P     ",
]


def get_game_art(type):  # 通过type找到相应的地图形式
    """Return game art based on type.

     Args:
       type: Type of game art.
             left => left open.
             right => right open.
             both => both left and right open.

    Returns:
       The game art based on type.

    Raises:
       ValueError if value not in ('left','right', 'both').
    """
    if type == "left":
        return LEFT_OPEN_GAME_ART
    if type == "right":
        return RIGHT_OPEN_GAME_ART
    if type == "both":
        return BOTH_OPEN_GAME_ART
    raise ValueError("type must be one of ['left','right', 'both']. Given: " + type)


class BlockedMaze(gym.Env):
    """Gym wrapper around BlockedMaze environment constructed in pycolab."""

    def __init__(self, game_type):
        """Init BlockedMaze environment.

        Args:
          game_type: Possible values ("left", "right", "both").
        """
        self.game_type = game_type
        self.action_space = range(4)
        self.observation_space = range(54)

        self.reset()

    def step(self, action):
        """Take given action and return the next state observation and reward."""
        obs, reward, gamma = self.game.play(action)
        return obs, reward, self.game.game_over, ""

    def reset(self):  # 保留思考
        """Resets the game and returns the observation for the start state."""
        self.game = ascii_art.ascii_art_to_game(
            get_game_art(self.game_type),
            what_lies_beneath=" ",
            sprites={"P": PlayerSprite},
        )
        obs, reward, gamma = self.game.its_showtime()
        self.intial_obs = obs

        return obs


def obsv2state(obs):  # 将矩阵拉平成一维之后返回player在一维数组的当前位置
    """Convert pycolab's observation to int state.
    The state is flattened in our case and is (0,54).

    Args:
      obs: Pycolab's Observation object.

    Returns:
      Integer state from (0,54).
    """
    state = np.array(obs.layers["P"], dtype=np.float).flatten()
    states = np.flatnonzero(state)
    assert len(states) == 1, "There should be just one P."
    return states[0]


import matplotlib.pyplot as plt
import numpy as np


def plot(x, y, xlabel, ylabel, legend, title):
    """Plots y vs x marking x-axis with xlabel and y-axis with ylabel.

    Args:
     x: 1-D array like data to be plotted on x-axis.
     y: 1-D array like data to be plotted on y-axis.
     xlabel: Label for x-axis.
     ylabel: Label for y-axis.
    """
    plt.plot(x, y, linewidth=3, color="red")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)
    plt.legend(
        legend, prop={"size": 16}, loc="upper left", fancybox=True, framealpha=0.5
    )
    plt.show()


from pycolab.examples.classics import four_rooms as env
from pycolab.rendering import ObservationToFeatureArray
import numpy as np
import random
import matplotlib.pyplot as plt

# Hyper parameters for expirements.
params = {
    "gamma": 0.95,
    "alpha": 0.1,
    "epsilon": 0.1,
    "episodes": 200,
    "runs": 10,
    "max_steps": 3000,
}

# Range of planning steps to try.
planning_nsteps = [0, 5, 50]  # dynaq中的planning的次数，取三个：0，5，50


class Model(object):  # 简单的环境模型,放入（s，a）生成（s',r）
    """Model class that returns next state, observation
    when presented with current state and action."""

    def __init__(self):
        """Init model class"""
        self.model = {}  # 此model为字典格式，输入(state, action)得到(next_state, reward)

    def set(
        self, state, action, next_state, reward
    ):  # 输入(state, action)则可以得到(next_state, reward)，将其对应为四元组（s,a,s',r）
        """Store (next_state, reward) information for given (state, action).

        Args:
          state: int-type value of current state.
          action: int-type value of current action.
          next_state: int-type next state returned by the environment when
                        action is presented.
          reward: float-type reward returned by the environment when action is
                    presented.
        """
        self.model[(state, action)] = (next_state, reward)

    def get(self, state, action):  # get表示提取原像的映射结果
        """Returns next_state, reward stored for (state, action).
        If no entry found for (state, action), it returns (None, None).
        This should never happen.

        Args:
          state: int-type value of the current state.
          action: int-type value of the action action.

        Returns:
          next_state: int-type value for the next_state most recently seen for
                        (state, action) pair.
          reward: float-type value for the reward most recently seen for
                        (state, action) pair.
        """
        return self.model.get((state, action), (None, None))

    def keys(self):  # keys返回键值，即第一列的（s，a）
        "Return all the (state, action) pair stored in model."
        return self.model.keys()

    def reset(self):
        "Clear the model. Delete all stored (state, action) pairs."
        self.model.clear()


# 一步直接Q-learning,n步planning
class DynaQ(object):
    """DynaQ implementation class. run(.) method runs the DynaQ algorithm."""

    def __init__(self, params, model, action_space, state_space, rng):
        """Init DynaQ object

        Args:
          params: Hyperparameters for the DynaQ model.
          model: Model of the world.
          action_spaces: 1D int array. Range of possible actions.
          state_space: 1D int array. Range of possible states.
          rng: Random Number Generator.
        """
        self.params = params
        self.model = model
        self.rng = rng

        self.action_space = action_space
        self.state_space = state_space

        # Q表示state 和action张成的reward矩阵
        self.Q = np.zeros((len(self.state_space), len(self.action_space)))

    def run(self, initial_state, maze, planning_nsteps=0):  # 完成一局游戏，并返回steps和rewards
        """Run DynaQ algorithm.

        Args:
           initial_state: Value on the start state.
           maze: Gym type object of one of the mazes.
           planning_nsteps: Number of times to do planning.
        """
        state = initial_state
        step = 0
        totalreward = 0
        done = False
        while not done:
            step += 1
            action = self._get_action(state)
            state, reward, done = self._run_step(state, action, maze, planning_nsteps)
            totalreward += reward
        return step, totalreward

    def _run_step(self, state, action, maze, planning_nsteps):
        """Run a single DynaQ step.
        This function samples the environment, updates the Q value,
        updates the model and then uses model to update Q value planning_nsteps
        time.

        Args:
          state: Current state on the world.
          action: Action to be taken.
          maze: Gym type object of one of the mazes.
          planning_nsteps: Number of times to do planning.
        """
        # 使用策略与真实环境进行一步Q learning(s情况下,采用a,得到s'和reward，然后进行Q-learning)

        # 真实迷宫中采用action，得到下一state和reward,并进行一步Q-learning
        obs, orig_reward, done, _ = maze.step(action)
        next_state = obsv2state(obs)
        self._updateQ(state, action, orig_reward, next_state)

        # 得到经验四元组（s,a,s',r）,保存并放入model中
        self.model.set(state, action, next_state, orig_reward)

        # Model Based: 使用得到的四元组，直接进行Q-planning，进行n次Q-planning
        for n in range(planning_nsteps):
            # 随机选择一个经验四元组
            obs_state_actions = self.model.keys()
            index = self.rng.choice(len(obs_state_actions))
            state, action = list(obs_state_actions)[index]
            new_state, reward = self.model.get(state, action)

            # 使用四元组进行Qlearning
            self._updateQ(state, action, reward, new_state)

        return next_state, orig_reward, done

    # 更新Q value
    def _updateQ(self, state, action, reward, next_state):
        """This functions updates the Q value.
          Q_{t+1}(s,a) = Q_t{s,a} + \alpha * (r + \gamma * max_a{Q_t(s,a)} -Q_t(s,a))

        Args:

          state: Current state of the world.
          action: Current action chosen.
          reward: Reward for taking action in state.
          next_state: The next state to transition to.
        """
        #注意，此处用了max(self.Q[next_state, :])，即下一个状态的Q值来更新当前状态的Q值
        #所以，当完成一局游戏之后，终点的前一个位置的Q值得到了更新
        #再继续玩多局游戏，可以继续更新反向更新路径上点的Q值
        tmp1 = self.Q[state, action]
        tmp2 = self.Q[next_state, :]
        self.Q[state, action] = self.Q[state, action] + self.params["alpha"] * (
            reward
            + self.params["gamma"] * np.max(self.Q[next_state, :])
            - self.Q[state, action]
        )
        if self.Q[state, action] > 0.0:
            print(1)

    def _get_action(self, state):
        """Choose action for state with a \epsilon-greedy policy.

        Args:
          state: Current state of the world.
        """
        # 鼓励策略探索
        if self.rng.uniform() < self.params["epsilon"]:
            return self.rng.choice(self.action_space)

        # 不探索情况下选择最优q_vals的action
        q_vals = self.Q[state]
        q_max = np.max(q_vals)

        # 先找出val值最大的action的位置，然后从最大位置中随机选取一个位置
        return self.rng.choice(
            [action for action, value in enumerate(q_vals) if value == q_max]
        )

    def reset(self):
        """Reset the DynaQ algorithms.
        This resets the model and resets the Q values.
        """
        self.model.reset()
        self.Q.fill(0)


model = Model()

left_open_maze = BlockedMaze("left")

legend = []

# 选择不同次数的planning，进行对比实验
for planning_nstep in planning_nsteps:
    print("Running Agent For Planning n-steps: %d" % planning_nstep)
    total_steps = np.zeros((params["runs"], params["episodes"]))

    # 训练run次，求均值来表示收敛的平均次数，以减少随机性影响
    for run in range(params["runs"]):
        # 在每一个大局run中，玩episode次游戏直到终点
        dynaQ = DynaQ(
            params,
            model,
            left_open_maze.action_space,
            left_open_maze.observation_space,
            np.random,
        )

        for episode in range(params["episodes"]):
            obs = left_open_maze.reset()
            initial_state = obsv2state(obs)
            steps, totalreward = dynaQ.run(
                initial_state, left_open_maze, planning_nstep
            )
            total_steps[run, episode] = steps
    episodes_index = np.linspace(
        0, params["episodes"], params["episodes"], endpoint=True
    )
    temp = "N Steps = {:}".format(planning_nstep)
    legend.append(temp)
    plt.plot(episodes_index, np.average(total_steps, axis=0), linewidth=3)

plt.xlabel("Episodes", fontsize=14)
plt.ylabel("Steps", fontsize=14)
plt.title("Average learning curves for Dyna-Q agents", fontsize=14)
plt.legend(legend, prop={"size": 16}, loc="upper right", fancybox=True, framealpha=0.5)
print(
    "The figure below shows the average learning curves for Dyna-Q agents varying in their number of planning steps (n) per real step."
)
print("           ")
print("           ")
plt.show()


#agent先学习left_open_maze，使用学习到的策略再学习right_open_maze
model = Model()

left_open_maze = BlockedMaze("left")
right_open_maze = BlockedMaze("right")

planning_nsteps = 10
print("For Planning n-steps: %d" % planning_nsteps)
print("")

params["alpha"] = 0.7
params["runs"] = 20
params["episodes"] = 1000
params["max_steps"] = 3000

rewards = np.zeros((params["runs"], params["max_steps"]))

for run in range(params["runs"]):
    cumulativereward = 0
    timesteps = 0
    last_steps = 0
    dynaQ = DynaQ(
        params,
        model,
        left_open_maze.action_space,
        left_open_maze.observation_space,
        np.random.RandomState(run),
    )
    for episode in range(params["episodes"]):
        if timesteps <= 1000:
            maze = right_open_maze
        elif timesteps > 1000:
            maze = left_open_maze

        obs = maze.reset()
        initial_state = obsv2state(obs)
        steps, totalreward = dynaQ.run(initial_state, maze, planning_nsteps)
        timesteps += steps

        if timesteps > params["max_steps"]:
            break

        cumulativereward += totalreward
        rewards[run, timesteps:] = cumulativereward


plot(
    range(0, params["max_steps"]),
    np.average(rewards, axis=0),
    "Time Steps",
    "Cumulative Reward",
    ["Dyna Q"],
    "Blocking Maze Task: Average cumulative reward for Dyna-Q agents",
)


#agent先学习left_open_maze，使用学习到的策略再学习both_open_maze
model = Model()

left_open_maze = BlockedMaze("left")
both_open_maze = BlockedMaze("both")

planning_nsteps = 50
print("For Planning n-steps: %d" % planning_nsteps)
print("       ")

params["alpha"] = 0.7
params["runs"] = 5
params["episodes"] = 1000
params["max_steps"] = 6000

rewards = np.zeros((params["runs"], params["max_steps"]))

for run in range(params["runs"]):
    cumulativereward = 0
    timesteps = 0
    last_steps = 0
    dynaQ = DynaQ(
        params,
        model,
        left_open_maze.action_space,
        left_open_maze.observation_space,
        np.random.RandomState(run),
    )
    for episode in range(params["episodes"]):
        if timesteps <= 3000:
            maze = left_open_maze
        elif timesteps > 3000:
            maze = both_open_maze

        obs = maze.reset()
        initial_state = obsv2state(obs)
        steps, totalreward = dynaQ.run(initial_state, maze, planning_nsteps)
        timesteps += steps

        if timesteps > params["max_steps"]:
            break

        cumulativereward += totalreward
        rewards[run, timesteps:] = cumulativereward


plot(
    range(0, params["max_steps"]),
    np.average(rewards, axis=0),
    "Time Steps",
    "Cumulative Reward",
    ["Dyna Q"],
    "Shortcut Maze Task: Average cumulative reward for Dyna-Q agents",
)
