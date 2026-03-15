import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization


from datetime import datetime

from color_segmentation import cd_color_segmentation

from cv_test import  test_algorithm

trials_directory = os.getcwd()+"/trials"

def run_test(hsv_range = None, filter_specs = None, margins = None):
    cone_csv_path = "./test_images_cone/test_images_cone.csv"
    cone_template_path = './test_images_cone/cone_template.png'

    scores = test_algorithm(cd_color_segmentation, cone_csv_path, cone_template_path, range_param = hsv_range, filter_param = filter_specs, margin_param = margins)

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

    columns_lis = ["Iteration #", "Avg", "Min", "Target"]
    pd.DataFrame(columns = columns_lis).to_csv(csv_path, mode="a",index=False)

    trial_data = { #stores best data of current trial
        "range": None,
        "filter_specs": None,
        "avg": 0,
        "min": 0,
        "target": 0,
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

            new_target = (new_avg + new_min) / 2

            if new_target > trial_data["target"]:
            # if new_avg > trial_data["avg"]:
                trial_data["range"] = new_distance_range
                trial_data["filter_specs"] = new_filter_specs
                trial_data["avg"] = new_avg
                trial_data["min"] = new_min
                trial_data["target"] = new_target
                trial_data["params"] = (new_distance_range, new_filter_specs)
                

                with open(txt_path, "a") as file:
                    file.write(create_trial_record(iter = i, data = trial_data))

                pd.DataFrame([[i, trial_data["avg"], trial_data["min"], trial_data["target"]]], columns = columns_lis).to_csv(csv_path, mode="a", index=False, header = False)
    
    return f"Avg: {trial_data["avg"]}\nMin: {trial_data["min"]}\nTarget: {trial_data["target"]}\nParams: {trial_data["params"]}"


def bayesian_func(hue_low, hue_high, sat_low, sat_high, val_low, val_high, 
                  x_margin, y_margin, switch_1, switch_2, switch_3, switch_4, switch_5, switch_6, 
                  size_1, size_2, size_3, size_4, size_5, size_6,
                  iter_1, iter_2, iter_3, iter_4, iter_5, iter_6):
    
    hsv_range = {
        "lower" : (hue_low, sat_low, val_low), 
        "upper" : (hue_high, sat_high, val_high)
    }

    filter_specs = {
        "switch": np.array([switch_1, switch_2, switch_3, switch_4, switch_5, switch_6], dtype = int),
        "sizes": np.array([size_1, size_2, size_3, size_4, size_5, size_6], dtype = int),
        "iterations": np.array([iter_1, iter_2, iter_3, iter_4, iter_5, iter_6], dtype = int)
    }
    margins = np.array((x_margin, y_margin), dtype=int)
    new_score = run_test(hsv_range = hsv_range, margins = margins)

    return {"hsv_range": hsv_range,
            "filter_specs": filter_specs,
            "margins": margins,
            "score" : new_score
            }
    

def run_trial_bayesian(num_iterations, trial_name, starting_point = None):
    trial_data = { #stores best data of current trial
        "range": None,
        "filter_specs": None,
        "margins": None,
        "avg": 0,
        "min": 0,
        "target": 0,
        "params": None
    }

    param_bounds = {
        'hue_low': (0.0, 180.0), 
        'hue_high': (0.0, 180.0),
        'sat_low': (0.0, 255.0),
        'sat_high': (0.0, 255.0),
        'val_low': (0.0, 255.0),    
        'val_high': (0.0, 255.0),
        "x_margin": (0.0, 20.0),
        "y_margin": (0.0, 50.0)

    }

    max_filters = 6
    switch_bounds = (0.0, 1.999)
    size_bounds = (2.0, 6.0)
    iter_bounds = (1.0, 4.0)


    for i in range(1, max_filters+1):
        if i == 2 or i == 3: #only turn on the second switch, so only use one dilation filter with size 5 and 2 iterations
            param_bounds[f"switch_{i}"] = (1,1.99)
            param_bounds[f"size_{i}"] = (6.0,6.99)
            param_bounds[f"iter_{i}"] = (4.0,4.99)
        else:
            param_bounds[f"switch_{i}"] = (0,0.99)
            param_bounds[f"size_{i}"] = size_bounds
            param_bounds[f"iter_{i}"] = iter_bounds

    # for i in range(1, max_filters+1):
    #     param_bounds[f"switch_{i}"] = (switch_bounds)
    #     param_bounds[f"size_{i}"] = size_bounds
    #     param_bounds[f"iter_{i}"] = iter_bounds

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

    columns_lis = ["Iteration #", "Avg", "Min", "Target"]
    pd.DataFrame(columns = columns_lis).to_csv(csv_path, mode="a",index=False)

    if starting_point is not None:
        starting_params, starting_target = starting_point
        optimizer.register(params = starting_params, target = starting_target)

    for i in range(num_iterations):
        new_params = optimizer.suggest()
        new_results = bayesian_func(**new_params)
        new_avg, new_min = new_results["score"]["avg"], new_results["score"]["min"]

        # new_target = (new_avg + new_min) / 2 - np.sqrt(np.abs(new_avg - new_min))
        new_target = (new_avg + new_min) / 2
        optimizer.register(params = new_params, target = new_target)

        if new_target > trial_data["target"]: #if there is a net increase
            trial_data["range"] = np.array(new_results["hsv_range"]).tolist()
            trial_data["filter_specs"]= np.array(new_results["filter_specs"]).tolist()
            trial_data["margins"] = np.array(new_results["margins"]).tolist()
            trial_data["avg"]= new_avg
            trial_data["min"] = new_min
            trial_data["target"] = new_target
            trial_data["params"] = new_params

            with open(txt_path, "a") as file:
                file.write(create_trial_record(iter = i, data = trial_data))

            pd.DataFrame([[i, trial_data["avg"], trial_data["min"], trial_data["target"]]], columns = columns_lis).to_csv(csv_path, mode="a", index=False, header = False)

    return f"Avg: {trial_data["avg"]}\nMin: {trial_data["min"]}\nTarget: {trial_data["target"]}\nParams: {trial_data["params"]}"

def create_trial_record(iter, data):
    record_msg = f"""Iteration {iter+1}:
    Ranges: {data["range"]}
    Avg: {data["avg"]}
    Min: {data["min"]}
    Target: {data["target"]}
    Filter Specs: {data["filter_specs"]}
    Margins: {data["margins"]}
    Params: {data["params"]}

"""
    return record_msg

def plot_trial(trial_name, show = True, save = True):
    trial_csv_path = trials_directory+"/"+trial_name+"/"+trial_name+".csv"
    df = pd.read_csv(trial_csv_path)

    ax = df.plot(kind="line", x="Iteration #", y=["Avg", "Min", "Target"], title=trial_name, ylim=(0.0,1.0))
    ax.scatter(df["Iteration #"], df["Avg"])
    ax.scatter(df["Iteration #"], df["Min"])
    ax.scatter(df["Iteration #"], df["Target"])

    if save:
        trial_save_path = trials_directory+"/"+trial_name+"/"+trial_name+".png"
        plt.savefig(trial_save_path)

    if show:
        plt.show()

def run_trials(num_trials, num_iterations, trial_name, starting_point = None, trial_type = "random", plot = True):
    print("Trials Started")
    trial_name = trial_type+"_"+trial_name+"_trial"
    for i in range(1, num_trials+1):
        new_trial_name = trial_name+"_"+str(i)
        if trial_type == "random":
            print(run_trial_random(num_iterations, new_trial_name, starting_point))
        elif trial_type == "bayesian":
            print(run_trial_bayesian(num_iterations, new_trial_name, starting_point))
        
        if plot:
            plot_trial(new_trial_name, show = False)

        print("Trials ",i," / ", num_trials," Completed")

if __name__ == '__main__':

    starting_params = {'hue_low': np.float64(0.0), 'hue_high': np.float64(139.60193939720276), 'sat_low': np.float64(232.20555183493386), 'sat_high': np.float64(253.7389425265687), 'val_low': np.float64(143.40429522983987), 'val_high': np.float64(253.09911711599446), 'x_margin': np.float64(1.2558736644414508), 'y_margin': np.float64(48.17037905581659), 'switch_1': np.float64(0.0), 'size_1': np.float64(2.0), 'iter_1': np.float64(1.0), 'switch_2': np.float64(1.999), 'size_2': np.float64(6.0), 'iter_2': np.float64(4.0), 'switch_3': np.float64(0.0), 'size_3': np.float64(6.0), 'iter_3': np.float64(1.0), 'switch_4': np.float64(0.0), 'size_4': np.float64(6.0), 'iter_4': np.float64(4.0), 'switch_5': np.float64(1.999), 'size_5': np.float64(6.0), 'iter_5': np.float64(4.0), 'switch_6': np.float64(0.0), 'size_6': np.float64(2.0), 'iter_6': np.float64(1.0)}
    starting_target =  0.8262840658496285

    starting_point = (starting_params, starting_target)

    run_trials(5, 200, "kingcrab", starting_point = starting_point, trial_type = "bayesian", plot = True)

    pass