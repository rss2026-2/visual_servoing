import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization

# from bayes_opt.logger import JSONLogger

from datetime import datetime

from color_segmentation import cd_color_segmentation

from cv_test import  test_algorithm

trials_directory = os.getcwd()+"/trials"

def run_test(distance_range, filter_specs):
    cone_csv_path = "./test_images_cone/test_images_cone.csv"
    cone_template_path = './test_images_cone/cone_template.png'

    scores = test_algorithm(cd_color_segmentation, cone_csv_path, cone_template_path, range_param = distance_range, filter_param = filter_specs)

    if scores:
        return {
            "avg" : np.mean(list(scores.values())),
            "min" : np.min(list(scores.values()))
        }

def run_trial_random(num_iterations, trial_name):
    trial_range = None
    trial_filter_specs = None
    trial_avg_score = 0
    trial_min_score = None

    trial_directory = trials_directory+"/"+trial_name
    os.makedirs(trial_directory, exist_ok = True)

    txt_path = trial_directory+"/"+trial_name+".txt"
    csv_path = trial_directory+"/"+trial_name+".csv"

    with open(txt_path, "a") as file:
        current_time = datetime.now().strftime("%I:%M:%S %p %h %d, %Y")
        file.write("[ "+trial_name+"] ("+current_time+")\nTotal Iterations: "+str(num_iterations)+"\n\n")

    columns_lis = ["Iteration #", "Avg", "Min"]
    pd.DataFrame(columns = columns_lis).to_csv(csv_path, mode="a",index=False)

    rng = np.random.default_rng()
    for i in range(num_iterations):
        new_distance_range = rng.random(size=(3,2))
        new_filter_specs = {
            "switch": rng.choice([0,1], size = 8),
            "sizes" : rng.integers(2, 8, size = 8),
            "iterations" : rng.choice([1,2,3,4,5], size = 8)
        }
        new_score = run_test(new_distance_range, new_filter_specs)

        if new_score: #if the distance range didn't fail on one of the tests
            new_avg, new_min = new_score["avg"], new_score["min"]
            if new_avg > trial_avg_score:
                trial_range = new_distance_range
                trial_filter_specs = new_filter_specs
                trial_avg_score = new_avg
                trial_min_score = new_min
                

         
                with open(txt_path, "a") as file:
                    file.write(create_trial_record(iter = i, range = trial_range, filter_specs = trial_filter_specs, avg = trial_avg_score, min = trial_min_score))

                pd.DataFrame([[i, trial_avg_score, trial_min_score]], columns = columns_lis).to_csv(csv_path, mode="a", index=False, header = False)


def bayesian_func(hue_low, hue_high, sat_low, sat_high, val_low, val_high, 
                  switch_1, switch_2, switch_3, switch_4, switch_5, switch_6, 
                  size_1, size_2, size_3, size_4, size_5, size_6,
                  iter_1, iter_2, iter_3, iter_4, iter_5, iter_6):
    
    distance_range = [(hue_low, hue_high), (sat_low, sat_high), (val_low, val_high)]
    filter_specs = {
        "switch": np.array([switch_1, switch_2, switch_3, switch_4, switch_5, switch_6], dtype = int),
        "sizes": np.array([size_1, size_2, size_3, size_4, size_5, size_6], dtype = int),
        "iterations": np.array([iter_1, iter_2, iter_3, iter_4, iter_5, iter_6], dtype = int)
    }
    new_score = run_test(distance_range, filter_specs)
    if new_score:
        return {"distance_range": distance_range,
                "filter_specs": filter_specs,
                "score" : new_score
                }
    

def run_trial_bayesian(num_iterations, trial_name):
    param_bounds = {
        'hue_low': (0.0, 1.0), 
        'hue_high': (0.0, 1.0),
        'sat_low': (0.0, 1.0),
        'sat_high': (1.0, 1.0),
        'val_low': (0.0, 1.0),    
        'val_high': (1.0, 1.0),
    }

    max_filters = 6
    size_bounds = (2.0, 8.0)
    iter_bounds = (1.0, 6.0)

    for i in range(1, max_filters+1):
        param_bounds[f"switch_{i}"] = (0.0, 2.0)
        param_bounds[f"size_{i}"] = size_bounds
        param_bounds[f"iter_{i}"] = iter_bounds

    # acquisition_function = acquisition.ExpectedImprovement(xi=0.0)
    optimizer = BayesianOptimization(
        f = bayesian_func,
        # acquisition_function=acquisition_function,
        pbounds = param_bounds,
        verbose = 1
    )

    trial_directory = trials_directory+"/"+trial_name
    os.makedirs(trial_directory, exist_ok = True)

    txt_path = trial_directory+"/"+trial_name+".txt"
    csv_path = trial_directory+"/"+trial_name+".csv"

    with open(txt_path, "a") as file:
        current_time = datetime.now().strftime("%I:%M:%S %p %h %d, %Y")
        file.write("[ "+trial_name+"] ("+current_time+")\nTotal Iterations: "+str(num_iterations)+"\n\n")

    columns_lis = ["Iteration #", "Avg", "Min"]
    pd.DataFrame(columns = columns_lis).to_csv(csv_path, mode="a",index=False)
    
    trial_range = None
    trial_filter_specs = None
    trial_avg_score = 0
    trial_min_score = None

    for i in range(num_iterations):
        new_params = optimizer.suggest()
        new_results = bayesian_func(**new_params)
        new_avg, new_min = new_results["score"]["avg"], new_results["score"]["min"]
        optimizer.register(params = new_params, target = new_avg)

        if new_avg > trial_avg_score:
            trial_range = np.array(new_results["distance_range"]).tolist()
            trial_filter_specs = np.array(new_results["filter_specs"]).tolist()
            trial_avg_score = new_avg
            trial_min_score = new_min

            with open(txt_path, "a") as file:
                file.write(create_trial_record(iter = i, range = trial_range, filter_specs = trial_filter_specs, avg = trial_avg_score, min = trial_min_score))

            pd.DataFrame([[i, trial_avg_score, trial_min_score]], columns = columns_lis).to_csv(csv_path, mode="a", index=False, header = False)

    # optimizer.probe(
    #     params =  {'hue_low': np.float64(0.3974032963932183), 'hue_high': np.float64(0.483277763345553), 'sat_low': np.float64(0.1995677462114977), 'sat_high': np.float64(1.0), 'val_low': np.float64(0.5490086482373446), 'val_high': np.float64(1.0)},
    #     lazy = True
    # )
    #| 35        | 0.8549207 | 0.8167585 | 0.7338955 | 0.1976128 | 1.0       | 0.2217446 | 1.0       | 0.7823677 | 6.0314428 | 5.1787788 | 1.0687893 | 5.0223946 | 4.2553392 | 1.4945291 | 4.7400167 | 3.1656142 | 0.9635839 | 8.0       | 1.7617630 | 0.0       | 5.1914982 | 1.1159275 | 0.0       | 8.0       | 4.1078981 
    #96        | 0.8942409 | 0.3352851 | 0.8509637 | 0.1847463 | 1.0       | 0.3965662 | 1.0       | 0.6730489 | 7.3115551 | 4.5004591 | 1.3942604 | 5.8895328 | 2.5250913 | 0.2060162 | 5.0110798 | 1.2222393 | 0.7157372 | 5.0243335 | 4.8877056 | 1.0967970 | 7.0311750 | 1.7833632 | 0.8650465 | 4.5582466 | 5.1768997 |
    # trial_directory = trials_directory+"/"+trial_name
    # os.makedirs(trial_directory, exist_ok = True)

    # json_path = trial_directory+"/logs.json"
    # logger = JSONLogger(json_path)
    # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

def create_trial_record(iter, range, filter_specs, avg, min):
    record_msg = f"""Iteration {iter}:
    Ranges:
        Hue: {range[0]}
        Saturation: {range[1]}
        Value: {range[2]}
    Avg: {avg}
    Min: {min}
    Filter Specs: {filter_specs}

"""
    return record_msg

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
    
def run_trials(num_trials, num_iterations, trial_type = "random", plot = True):
    print("Trials Started")
    for i in range(1, num_trials+1):
        trial_name = trial_type+"_"+"trial_"+str(i)

        if trial_type == "random":
            run_trial_random(num_iterations, trial_name)
        elif trial_type == "bayesian":
            run_trial_bayesian(num_iterations, trial_name)
        
        if plot:
            plot_trial(trial_name, show = False)

        print("Trials ",i," / ", num_trials)

if __name__ == '__main__':
    run_trials(
        num_trials = 3,
        num_iterations = 100,
        trial_type = "bayesian"
    )
    pass