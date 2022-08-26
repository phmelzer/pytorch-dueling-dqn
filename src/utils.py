import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import os


def plot_learning_curve(score_list, avg_score_list, env):
    plt.figure(figsize=(12, 8))
    plt.plot(np.array(score_list), label="Score")
    plt.plot(np.array(avg_score_list), label="Average Score")
    plt.ylabel("Score")
    plt.xlabel("Episodes")
    plt.grid(axis="both")
    plt.legend(loc="lower right")
    plt.title('Learning Curve')
    plt.savefig('../plots/{}_dueling_dqn_learning_curve.png'.format(env))
    plt.close()


def store_training_data(episodes, scores, avg_scores, epsilons, env):
    training_data_dict = {"episode": episodes, "score": scores, "avg_score": avg_scores, "epsilon": epsilons}
    df = pd.DataFrame(training_data_dict)
    df.to_csv("../data/training/{}_dueling_dqn_training_data.csv".format(env))


def store_training_config(config, env):
    df = pd.DataFrame(config, index=[0])
    df.to_csv("../data/training/{}_dueling_dqn_training_config.csv".format(env))


def load_config(config_name):
    with open(os.path.join(config_name)) as file:
        config = yaml.safe_load(file)
    return config


def load_logging_config(filename):
    with open(filename, "rt") as f:
        logging_config = yaml.safe_load(f.read())
        f.close()
    return logging_config
