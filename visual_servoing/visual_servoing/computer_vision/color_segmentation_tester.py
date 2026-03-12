import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization, acquisition

from datetime import datetime

from color_segmentation import cd_color_segmentation

from cv_test import  test_algorithm

trials_directory = os.getcwd()+"/trials"

def test_by_distance_range(distance_range):
    cone_csv_path = "./test_images_cone/test_images_cone.csv"
    cone_template_path = './test_images_cone/cone_template.png'

    scores = test_algorithm(cd_color_segmentation, cone_csv_path, cone_template_path, range_param = distance_range)

    if scores:
        return {
            "avg" : np.mean(list(scores.values())),
            "min" : np.min(list(scores.values()))
        }

def run_trial_random(num_iterations, trial_name, record_type = "jumps"):
    trial_range = None
    trial_avg_score = 0
    trial_min_score = None

    trial_directory = trials_directory+"/"+trial_name
    os.makedirs(trial_directory, exist_ok = True)

    txt_path = trial_directory+"/"+trial_name+".txt"
    csv_path = trial_directory+"/"+trial_name+".csv"

    with open(txt_path, "a") as file:
        current_time = datetime.now().strftime("%I:%M:%S %p %h %d, %Y")
        file.write("[ "+trial_name+"] ("+current_time+")\nTotal Iterations: "+str(num_iterations)+"\nRecord Type: "+record_type+"\n\n")

    columns_lis = ["Iteration #", "Avg", "Min"]
    pd.DataFrame(columns = columns_lis).to_csv(csv_path, mode="a",index=False)

    rng = np.random.default_rng()
    for i in range(num_iterations):
        new_distance_range = rng.random(size=(2,3))
        new_score = test_by_distance_range(new_distance_range)

        if new_score: #if the distance range didn't fail on one of the tests
            new_avg, new_min = new_score["avg"], new_score["min"]
            if new_avg > trial_avg_score and new_min != 0:
                trial_range = new_distance_range
                trial_avg_score = new_avg
                trial_min_score = new_min

                if record_type == "jumps":
                    with open(txt_path, "a") as file:
                        file.write("Iteration "+str(i)+":\n\tRange: "+str(trial_range)+"\n\tAvg: "+str(trial_avg_score)+"\n\tMin: "+str(trial_min_score)+"\n")

                    pd.DataFrame([[i, trial_avg_score, trial_min_score]], columns = columns_lis).to_csv(csv_path, mode="a", index=False, header = False)
        
        if record_type == "periodic" and i%(i//10) == 0:
            with open(txt_path, "a") as file:
                file.write("Iteration "+str(i)+":\n\tRange: "+str(trial_range)+"\n\tAvg: "+str(trial_avg_score)+"\n\tMin: "+str(trial_min_score)+"\n")
            
            pd.DataFrame([[i, trial_avg_score, trial_min_score]], columns = columns_lis).to_csv(csv_path, mode="a", index=False, header = False)


def bayesian_func(hue_low, hue_high, sat_low, sat_high, val_low, val_high):
    distance_range = [(hue_low, hue_high), (sat_low, sat_high), (val_low, val_high)]
    new_score = test_by_distance_range(distance_range)
    if new_score:
        return new_score["avg"]
    else:
        return 0

def run_trial_optimize(num_iterations, trial_name, record_type = "jumps"):
    param_bounds = {
        'hue_low': (0.0, 1.0), 
        'hue_high': (0.0, 1.0),
        'sat_low': (0.0, 1.0),
        'sat_high': (1.0, 1.0),
        'val_low': (0.0, 1.0),
        'val_high': (1.0, 1.0),
    }
    acquisition_function = acquisition.ExpectedImprovement(xi=0.0)
    optimizer = BayesianOptimization(
        f = bayesian_func,
        acquisition_function=acquisition_function,
        pbounds = param_bounds,
    )

    optimizer.probe(
        params =  {'hue_low': np.float64(0.3974032963932183), 'hue_high': np.float64(0.483277763345553), 'sat_low': np.float64(0.1995677462114977), 'sat_high': np.float64(1.0), 'val_low': np.float64(0.5490086482373446), 'val_high': np.float64(1.0)},
        lazy = True
    )

    optimizer.maximize(
        init_points = 10,
        n_iter = num_iterations,

    )

    return optimizer.max

def plot_trial(trial_name, show = True, save = True):
    trial_csv_path = trials_directory+"/"+trial_name+"/"+trial_name+".csv"
    df = pd.read_csv(trial_csv_path)

    ax = df.plot(kind="line", x="Iteration #", y=["Avg", "Min"], title=trial_name, ylim=(0.0,1.0))
    ax.scatter(df["Iteration #"], df["Avg"])
    ax.scatter(df["Iteration #"], df["Min"])

    if save:
        trial_save_path = trials_directory+"/"+trial_name+"/"+trial_name+".png"
        plt.savefig(trial_save_path)

    if show:
        plt.show()
    

if __name__ == '__main__':
    iterations = 100
    num_trials = 3
    # for i in range(1, num_trials+1):
    #     trial_name = "trial_"+str(i)
    #     run_trial_random(iterations, trial_name)
    #     plot_trial(trial_name, show=False)

    #     print("Trials ",i," / ", num_trials)

    print(run_trial_optimize(iterations, "trial_1"))
    pass