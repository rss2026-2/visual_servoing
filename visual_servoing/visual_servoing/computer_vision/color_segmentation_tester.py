import os
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime


trials_directory = os.getcwd()+"/trials"

def test_by_distance_range(distance_range):
    command = ["python", "cv_test.py", "cone", "color", *[str(val) for val in distance_range]]

    try:
        # Run the command, capture output, and use text=True for string output (Python 3.7+)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True  # Raise an exception if the command fails
        )

        # Read the output from the terminal (stdout and stderr)
        avg_score, min_score = [float(val) for val in result.stdout.split(",")]
        return {"avg":avg_score, "min":min_score}

    except subprocess.CalledProcessError as e:
        pass
    except FileNotFoundError:
        pass

def run_trial(num_iterations, trial_name, record_type = "jumps"):
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
        new_distance_range = rng.random(size=3)
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
    iterations = 1000
    num_trials = 10
    for i in range(1, num_trials+1):
        trial_name = "trial_"+str(i)
        run_trial(iterations, trial_name)
        plot_trial(trial_name, show=False)

        print("Trials ",i," / ", num_trials)

    pass