""" Train Agent """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tf info, warning and error messages are not printed
import gym
import logging.config
import numpy as np
from agent import Agent
import utils
from time import sleep

config = utils.load_config("../config/config.yaml")
logging_config = utils.load_logging_config("../config/logging.yaml")


def train():
    env = gym.make(config["env_name"])
    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, lr=config["learning_rate"],
                     discount_factor=config["discount_factor"], eps=config["eps"], eps_dec=config["eps_dec"],
                     eps_min=config["eps_min"], batch_size=config["batch_size"],
                  replace=config["replace_target_network_cntr"], mem_size=config["mem_size"], algo="dueling_dqn",
                     env_name=config["env_name"])

    # lists for storing data
    episode_list = []
    score_list = []
    avg_score_list = []
    epsilon_list = []
    best_score = -np.inf

    logger.info("Start training")
    for episode in range(config["training_episodes"]):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            if config["render"]:
                env.render()
                sleep(0.1)
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()

        episode_list.append(episode)
        score_list.append(score)
        epsilon_list.append(agent.eps)
        avg_score = np.mean(score_list[-100:])
        avg_score_list.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score

        logger.info('episode: {}, score: {}, avg_score: {}, best_score: {}, epsilon: {}'.format(episode, "%.2f" % score,
                                                                                                "%.2f" % avg_score,
                                                                                                "%.2f" % best_score,
                                                                                                "%.2f" % agent.eps))
    # save policy and target network
    agent.save_models()
    logger.info("Save models")

    logger.info("Finish training")

    # plot simple learning curve
    utils.plot_learning_curve(score_list, avg_score_list, config["env_name"])

    # store training data and config to csv file
    utils.store_training_data(episode_list, score_list, avg_score_list, epsilon_list, config["env_name"])
    utils.store_training_config(config, config["env_name"])


if __name__ == "__main__":
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger("train")
    train()
