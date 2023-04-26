import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("time_file")
    args = parser.parse_args()
    with open(args.time_file, "r") as f:
        time = [float(t.strip()) for t in f.readlines()]
        serial_time = time[::2]
        thread_time = time[1::2]
        serial_time = min(serial_time)
        time = [serial_time] + thread_time
    x = list(range(1, 5))
    X_Y_Spline = make_interp_spline(x, time)
    X = np.linspace(1, 4, 100)
    Y = X_Y_Spline(X)
    plt.plot(X, Y, zorder=0)
    plt.xlabel("num thread")
    plt.ylabel("time (ms)")
    plt.scatter(x, time, marker='o', c='black', zorder=1)
    plt.title(f"{args.time_file[4:-9]}_Time-Threads")
    plt.savefig(f"{args.time_file[:-9]}_TT.png")
