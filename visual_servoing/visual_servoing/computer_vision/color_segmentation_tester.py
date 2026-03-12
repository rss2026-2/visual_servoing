import subprocess
import numpy as np
import os
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

    
    os.makedirs(trials_directory, exist_ok = True)
    trial_path = trials_directory+"/"+trial_name+".txt"

    with open(trial_path, "a") as file:
        current_time = datetime.now().strftime("%I:%M:%S %p %h %d, %Y")
        file.write("[ "+trial_name+"] ("+current_time+")\nTotal Iterations: "+str(num_iterations)+"\nRecord Type: "+record_type+"\n\n")

    rng = np.random.default_rng()
    for i in range(num_iterations):
        new_distance_range = rng.random(size=3)
        new_score = test_by_distance_range(new_distance_range)

        if new_score: #if the distance range didn't fail on one of the tests
            new_avg, new_min = new_score["avg"], new_score["min"]
            if new_avg > trial_avg_score:
                trial_range = new_distance_range
                trial_avg_score = new_avg
                trial_min_score = new_min

                if record_type == "jumps":
                    with open(trial_path, "a") as file:
                        file.write("Iteration "+str(i)+":\n\tRange: "+str(trial_range)+"\n\tAvg: "+str(trial_avg_score)+"\n\tMin: "+str(trial_min_score)+"\n")
        
        if record_type == "periodic" and i%(i//10) == 0:
            with open(trial_path, "a") as file:
                file.write("Iteration "+str(i)+":\n\tRange: "+str(trial_range)+"\n\tAvg: "+str(trial_avg_score)+"\n\tMin: "+str(trial_min_score)+"\n")

    
    with open(trial_path, "a") as file:
        file.write("\nFinal Results:\n\tRange: "+str(trial_range)+"\n\tAvg: "+str(trial_avg_score)+"\n\tMin: "+str(trial_min_score)+"\n")


if __name__ == '__main__':

    # for i in range(3, 13):
    #     run_trial(1000, "trial_"+str(i))

    # run_trial(5000, "trial_13")
    pass