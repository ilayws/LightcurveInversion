import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ray_tracer import n_files, n_samples, n_rots

# You can run this to test how the lightcurves you generated look

data = pd.read_csv("data.csv", header=None)
data = np.array(data)

avg_light = []
i = np.random.randint(0,n_samples*n_files)
angledata = np.linspace(0,2*np.pi,n_rots)
for i in range(5):
    
    lightdata = data[i,-n_rots:]
    avg_light.append( np.trapz(lightdata, angledata) / (2*np.pi))
    print(f"Sample: {i}")
    print(avg_light[-1])
    plt.plot(angledata,lightdata)
    plt.plot([0,2*np.pi], [avg_light[-1],avg_light[-1]])
    plt.xlabel("Angle (π rad)")
    plt.ylabel("Light reflected")
    plt.show()