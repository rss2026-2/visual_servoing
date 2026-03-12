import subprocess
import numpy as np



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


num_trials = 1000
best_range = None
best_avg = 0
lowest_min = None

rng = np.random.default_rng()
for i in range(num_trials):
    new_range = rng.random(size=3)

    new_score = test_by_distance_range(new_range)
    if new_score:
        new_avg, new_min = new_score["avg"], new_score["min"]

        if new_avg > best_avg:
            best_avg = new_avg
            best_range = new_range
            lowest_min = new_min
    
    if i%(num_trials//10) == 0:
        print("Trials: ",i," / ",num_trials)
        print("Current best avg: ",best_avg, " Current best range: ", best_range, "Min score: ",lowest_min)
        
print("\nFinal Results: Best Average: ",best_avg," Best Range: ", best_range, " Min Score: ",lowest_min)

#0.8786302510610119
# [0.40085958 0.17393698 0.57671796]

#0.8837653619188384
#0.7497371188222923
#[0.88133733 0.21231441 0.58131598]

# 0.8794207065330817
# [0.5070381  0.17759026 0.55766663]

#Current best avg:  0.8823312338722543  Current best range:  [0.86659887 0.19177398 0.59479711] Min score:  0.773109243697479

#plot the jumps in avg and the trials where they happened
#overfitting to training data and being less reliable in real life