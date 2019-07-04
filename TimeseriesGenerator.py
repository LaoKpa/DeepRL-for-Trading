import numpy as np

# function to generate set of timeseries and labels
def make_timeseries(win, num_samples):
    x_arr = []
    y_arr = []
    # build batch
    # randomly generate linear positive or negative functions and assign label
    for i in range(num_samples):
        upDown = np.random.rand()
        rand = np.random.randint(0, 1000)
        if upDown > 0.5:

            x = np.linspace(rand, rand + win, num=win)
            x_arr.append(x)
            y = 1
            y_arr.append(y)

        else:
            x = np.linspace(rand + win, rand, num=win)
            x_arr.append(x)
            y = 0
            y_arr.append(y)

    # convert to array for training input
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    return x_arr, y_arr




