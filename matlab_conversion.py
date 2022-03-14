import mne
import scipy.io
import os
import glob
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

dataset_dir = "/mnt/cifs/rdss/rdss_kahwang/Shared StopOR data"
LFP_dir = os.path.join(dataset_dir, "LFP pp/")

# for file in glob.glob(LFP_dir + "ManualRej???.mat"):
#     subject_rejections = scipy.io.loadmat(file)
#     print(subject_rejections)

array_labels = [
    "setname",
    "filename",
    "filepath",
    "subject",
    "group",
    "condition",
    "session",
    "comments",
    "nbchan",
    "trials",
    "pnts",
    "srate",
    "xmin",
    "xmax",
    "times",
    "data",
    "icaact",
    "icawinv",
    "icasphere",
    "icaweights",
    "icachansind",
    "chanlocs",
    "urchanlocs",
    "chaninfo",
    "ref",
    "event",
    "urevent",
    "eventdescription",
    "epoch",
    "epochdescription",
    "reject",
    "stats",
    "specdata",
    "specicaact",
    "splinefile",
    "icasplinefile",
    "dipfit",
    "history",
    "saved",
    "etc",
]

event_labels = [
    "type",
    "duration",
    "latency",
    "tnum",
    "side",
    "resp_side",
    "acc",
    "rt",
    "behav",
]
for file in glob.glob(LFP_dir + "Subject???-LFP.mat"):
    subject_data = scipy.io.loadmat(file)
    LFP_data = subject_data["LFP"][0][0]
    lfp_dict = {}
    for i, array_item in enumerate(LFP_data):
        array_label = array_labels[i]

        try:
            if array_label == "data":
                item = array_item
            else:
                item = array_item[0]
        except:
            lfp_dict[array_label] = None
            continue

        if array_label == "event":
            df = pd.DataFrame(item, columns=event_labels)
            item = df

        lfp_dict[array_label] = item

    # create mevent annotations
    event_df = lfp_dict["event"]
    events = np.zeros([len(event_df.index), 3])

    stimulus_mapping = {"GoStimulus": 1, "StopStimulus": 2, "Response": 3}
    for i, event in event_df.iterrows():
        events[i, 0] = round(event["latency"][0][0])
        events[i, 2] = stimulus_mapping.get(event["type"][0])

    event_mapping = {1: "GoStimulus", 2: "StopSimulus", 3: "Response"}
    annot_from_events = mne.annotations_from_events(
        events=events,
        event_desc=event_mapping,
        sfreq=lfp_dict["srate"][0],
    )

    # SG = subgaleal
    channel_locs = [
        loc[0][0].replace("subgaleal", "SG") for loc in lfp_dict["chanlocs"]
    ]

    info = mne.create_info(ch_names=channel_locs, sfreq=lfp_dict["srate"][0])
    data = lfp_dict["data"]
    channels = mne.io.RawArray(np.array(data), info)
    channels.set_annotations(annot_from_events)
    channels.plot(start=0, duration=5)
    plt.show()

    ### todo remove manual rejections
