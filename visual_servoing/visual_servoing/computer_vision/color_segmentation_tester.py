import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization


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
    trial_directory = trials_directory+"/"+trial_name
    os.makedirs(trial_directory, exist_ok = True)

    txt_path = trial_directory+"/"+trial_name+".txt"
    csv_path = trial_directory+"/"+trial_name+".csv"

    with open(txt_path, "a") as file:
        current_time = datetime.now().strftime("%I:%M:%S %p %h %d, %Y")
        file.write("[ "+trial_name+"] ("+current_time+")\nTotal Iterations: "+str(num_iterations)+"\n\n")

    columns_lis = ["Iteration #", "Avg", "Min", "Avg_Min_Mean"]
    pd.DataFrame(columns = columns_lis).to_csv(csv_path, mode="a",index=False)

    trial_data = { #stores best data of current trial
        "range": None,
        "filter_specs": None,
        "avg": 0,
        "min": 0,
        "avg_min_mean": 0,
        "params": None
    }

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

            new_avg_min_mean = (new_avg + new_min) / 2

            if new_avg_min_mean > trial_data["avg_min_mean"]:
            # if new_avg > trial_data["avg"]:
                trial_data["range"] = new_distance_range
                trial_data["filter_specs"] = new_filter_specs
                trial_data["avg"] = new_avg
                trial_data["min"] = new_min
                trial_data["avg_min_mean"] = new_avg_min_mean
                trial_data["params"] = (new_distance_range, new_filter_specs)
                

                with open(txt_path, "a") as file:
                    file.write(create_trial_record(iter = i, data = trial_data))

                pd.DataFrame([[i, trial_data["avg"], trial_data["min"], trial_data["avg_min_mean"]]], columns = columns_lis).to_csv(csv_path, mode="a", index=False, header = False)
    
    return f"Avg: {trial_data["avg"]}\nMin: {trial_data["min"]}\nAvg_Min_Mean: {trial_data["avg_min_mean"]}\nParams: {trial_data["params"]}"


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

    return {"distance_range": distance_range,
            "filter_specs": filter_specs,
            "score" : new_score
            }

    

def run_trial_bayesian(num_iterations, trial_name):
    trial_data = { #stores best data of current trial
        "range": None,
        "filter_specs": None,
        "avg": 0,
        "min": 0.0,
        "params": None
    }

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
        if i == 2:
            param_bounds[f"switch_{i}"] = (1,1.25)
            param_bounds[f"size_{i}"] = (4.0,4.25)
            param_bounds[f"iter_{i}"] = (1.0,2.99)
        else:
            param_bounds[f"switch_{i}"] = (0,0.5)
            param_bounds[f"size_{i}"] = size_bounds
            param_bounds[f"iter_{i}"] = iter_bounds

    # acquisition_function = acquisition.ExpectedImprovement(xi=0.0)
    optimizer = BayesianOptimization(
        f = bayesian_func,
        # acquisition_function=acquisition_function,
        pbounds = param_bounds,
        verbose = 2
    )

    trial_directory = trials_directory+"/"+trial_name
    os.makedirs(trial_directory, exist_ok = True)

    txt_path = trial_directory+"/"+trial_name+".txt"
    csv_path = trial_directory+"/"+trial_name+".csv"

    with open(txt_path, "a") as file:
        current_time = datetime.now().strftime("%I:%M:%S %p %h %d, %Y")
        file.write("[ "+trial_name+ "] ("+current_time+")\nTotal Iterations: "+str(num_iterations)+"\n\n")

    columns_lis = ["Iteration #", "Avg", "Min"]
    pd.DataFrame(columns = columns_lis).to_csv(csv_path, mode="a",index=False)

    # | 12        | 0.8903852 | 0.8216992 | 1.0       | 0.1553830 | 1.0       | 0.4302919 | 1.0       | 0.1272110 | 4.5022429 | 2.1376954 | 1.6276444 | 3.1547204 | 1.4919597 | 0.5449078 | 6.3001111 | 1.1509766 | 0.9768408 | 4.0874484 | 2.7310682 | 0.6673591 | 4.6589857 | 4.2833414 | 0.1877486 | 6.1806158 | 1.4588511 |
    starting_point = {'hue_low': np.float64(0.26708526688793505), 'hue_high': np.float64(0.8421780462961072), 'sat_low': np.float64(0.15168815426382168), 'sat_high': np.float64(1.0), 'val_low': np.float64(0.4450195301618705), 'val_high': np.float64(1.0), 'switch_1': np.float64(0.19736675015654012), 'size_1': np.float64(5.401711360055775), 'iter_1': np.float64(1.5017263013699773), 'switch_2': np.float64(1.6778872383401953), 'size_2': np.float64(4.490137140266256), 'iter_2': np.float64(1.0927518185930578), 'switch_3': np.float64(0.8924516233867578), 'size_3': np.float64(5.81425090360953), 'iter_3': np.float64(1.0), 'switch_4': np.float64(0.8808059906878062), 'size_4': np.float64(3.939185464183444), 'iter_4': np.float64(2.0818692891735138), 'switch_5': np.float64(0.7167230310191588), 'size_5': np.float64(4.430389205765636), 'iter_5': np.float64(4.26393011668932), 'switch_6': np.float64(0.8881518410658341), 'size_6': np.float64(5.459211480331205), 'iter_6': np.float64(2.385359317018129)}
    starting_target = 0.8936774856127421

    optimizer.register(params = starting_point, target = starting_target)

    for i in range(num_iterations):
        new_params = optimizer.suggest()
        new_results = bayesian_func(**new_params)
        new_avg, new_min = new_results["score"]["avg"], new_results["score"]["min"]
        optimizer.register(params = new_params, target = new_avg)

        if (new_avg - trial_data["avg"]) + (new_min - trial_data["min"]) > 0: #if there is a net increase
            trial_data["range"] = np.array(new_results["distance_range"]).tolist()
            trial_data["filter_specs"]= np.array(new_results["filter_specs"]).tolist()
            trial_data["avg"]= new_avg
            trial_data["min"] = new_min
            trial_data["params"] = new_params

            with open(txt_path, "a") as file:
                file.write(create_trial_record(iter = i, data = trial_data))

            pd.DataFrame([[i, trial_data["avg"], trial_data["min"]]], columns = columns_lis).to_csv(csv_path, mode="a", index=False, header = False)

    return f"Avg: {trial_data["avg"]}\nMin: {trial_data["min"]}\nParams: {trial_data["params"]}"

def create_trial_record(iter, data):
    record_msg = f"""Iteration {iter}:
    Ranges:
        Hue: {data["range"][0]}
        Saturation: {data["range"][1]}
        Value: {data["range"][2]}
    Avg: {data["avg"]}
    Min: {data["min"]}
    Avg_Min_Mean: {data["avg_min_mean"]}
    Filter Specs: {data["filter_specs"]}

"""
    return record_msg

def plot_trial(trial_name, show = True, save = True):
    trial_csv_path = trials_directory+"/"+trial_name+"/"+trial_name+".csv"
    df = pd.read_csv(trial_csv_path)

    ax = df.plot(kind="line", x="Iteration #", y=["Avg", "Min", "Avg_Min_Mean"], title=trial_name, ylim=(0.0,1.0))
    ax.scatter(df["Iteration #"], df["Avg"])
    ax.scatter(df["Iteration #"], df["Min"])
    ax.scatter(df["Iteration #"], df["Avg_Min_Mean"])

    if save:
        trial_save_path = trials_directory+"/"+trial_name+"/"+trial_name+".png"
        plt.savefig(trial_save_path)

    if show:
        plt.show()

def run_trials(num_trials, num_iterations, trial_name, trial_type = "random", plot = True):
    print("Trials Started")

    for i in range(1, num_trials+1):
        trial_name = trial_type+"_"+trial_name+"_trial_"+str(i)

        if trial_type == "random":
            print(run_trial_random(num_iterations, trial_name))
        elif trial_type == "bayesian":
            print(run_trial_bayesian(num_iterations, trial_name))
        
        if plot:
            plot_trial(trial_name, show = False)

        print("Trials ",i," / ", num_trials," Completed")

if __name__ == '__main__':
    # run_trials(
    #     num_trials = 6,
    #     num_iterations = 300,
    #     trial_type = "bayesian"
    # )

    run_trials(1, 1000, "yoshi", "random", True)
    pass