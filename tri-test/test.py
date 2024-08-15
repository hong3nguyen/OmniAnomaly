import os
from pathlib import Path
from typing import List

import numpy as np 
import pandas as pd 
# from sklearn.ensemble import IsolationForest
# from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
# from urllib2 import urlopen
import urllib.request

import temporian as tp 

#
# Download DATAset
#
# machines_per_group = [8, 9, 11]  # Full dataset
machines_per_group = [3, 3, 3]  # Subset of data

machines = [f"machine-{group}-{id}" 
    for group, machine in zip(range(1, 4), machines_per_group) for id in range(1, machine + 1)]
print("{len(machines)} machines")

data_dir = Path("tmp/temporian_server_machine_dataset")
dataset_url = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"

data_dir.mkdir(parents=True, exist_ok=True)

# Download the data and labels for each machine to its own folder
for machine in machines:
    print(f"Download data of {machine}")

    machine_dir = data_dir / machine
    machine_dir.mkdir(exist_ok=True)

    data_path = machine_dir / "data.csv"
    if not data_path.exists():
        urllib.request.urlretrieve(f"{dataset_url}/test/{machine}.txt", data_path)

    labels_path = machine_dir / "labels.csv"
    if not labels_path.exists():
         urllib.request.urlretrieve(f"{dataset_url}/test_label/{machine}.txt", labels_path)

#
# LOAD DATA 
#``
dataframes = []

for machine in machines:
    machine_dir = data_dir / machine

    # Read the data and labels
    print(f"Load data of {machine}...", end="")
    df = pd.read_csv(machine_dir / "data.csv", header=None).add_prefix("f")
    labels = pd.read_csv(machine_dir/ "labels.csv", header=None)
    df = df.assign(label=labels)
    df["machine"] = machine
    df["timestamp"] = range(df.shape[0])
    print(f"found {df.shape[0]} events")

    dataframes.append(df)

dataframes[0].head(3)

# Convert the dataframes into a single Temporian EventSet

evset = tp.combine(*map(tp.from_pandas, dataframes))

# Index the EventSet according the the machine name.
evset = evset.set_index("machine")

# Cast the feature and label to a smaller dtypes to same one memory.
evset = evset.cast(tp.float32).cast({"label": tp.int32})


import matplotlib.pyplot as plt

# Plot the first 3 features
dataFeature = evset.plot(indexes="machine-1-1", max_num_plots=3)
plt.savefig('dataFeature.png', bbox_inches='tight')

# Plot the labels
ax = evset["label"].plot(indexes="machine-1-1")
#ax.set_title('Labels')

# Save the figure as an image
plt.savefig('labels.png', bbox_inches='tight')  

# Display the plot
plt.show()

print(evset)

