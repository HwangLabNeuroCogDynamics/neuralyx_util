# script to test how to read neuralyx data
# modififed from code kindly provided by
# alafuzof (https://github.com/alafuzof)
# https://github.com/alafuzof/NeuralynxIO

import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import numpy as np
import datetime
import glob as glob
import mne
import statistics
import pandas as pd

HEADER_LENGTH = 16 * 1024  # 16 kilobytes of header

NCS_SAMPLES_PER_RECORD = 512
NCS_RECORD = np.dtype(
    [
        (
            "TimeStamp",
            np.uint64,
        ),  # Cheetah timestamp for this record. This corresponds to
        # the sample time for the first data point in the Samples
        # array. This value is in microseconds.
        (
            "ChannelNumber",
            np.uint32,
        ),  # The channel number for this record. This is NOT the A/D
        # channel number
        (
            "SampleFreq",
            np.uint32,
        ),  # The sampling frequency (Hz) for the data stored in the
        # Samples Field in this record
        (
            "NumValidSamples",
            np.uint32,
        ),  # Number of values in Samples containing valid data
        ("Samples", np.int16, NCS_SAMPLES_PER_RECORD),
    ]
)  # Data points for this record. Cheetah
# currently supports 512 data points per
# record. At this time, the Samples
# array is a [512] array.

NEV_RECORD = np.dtype(
    [
        ("stx", np.int16),  # Reserved
        ("pkt_id", np.int16),  # ID for the originating system of this packet
        ("pkt_data_size", np.int16),  # This value should always be two (2)
        ("TimeStamp", np.uint64),  # Cheetah timestamp for this record. This value is in
        # microseconds.
        ("event_id", np.int16),  # ID value for this event
        ("ttl", np.int16),  # Decimal TTL value read from the TTL input port
        ("crc", np.int16),  # Record CRC check from Cheetah. Not used in consumer
        # applications.
        ("dummy1", np.int16),  # Reserved
        ("dummy2", np.int16),  # Reserved
        (
            "Extra",
            np.int32,
            8,
        ),  # Extra bit values for this event. This array has a fixed
        # length of eight (8)
        ("EventString", "S", 128),
    ]
)  # Event string associated with this event record. This string
# consists of 127 characters plus the required null termination
# character. If the string is less than 127 characters, the
# remainder of the characters will be null.

VOLT_SCALING = (1, "V")
MILLIVOLT_SCALING = (1000, "mV")
MICROVOLT_SCALING = (1000000, "µV")


def read_header(fid):
    # Read the raw header data (16 kb) from the file object fid. Restores the position in the file object after reading.
    pos = fid.tell()
    fid.seek(0)
    raw_header = fid.read(HEADER_LENGTH).strip(b"\0")
    fid.seek(pos)

    return raw_header


def parse_header(raw_header):
    # Parse the header string into a dictionary of name value pairs
    header = dict()

    # Decode the header as iso-8859-1 (the spec says ASCII, but there is at least one case of 0xB5 in some headers)
    raw_header = raw_header.decode("iso-8859-1")

    # Neuralynx headers seem to start with a line identifying the file, so
    # let's check for it
    header_lines = [line.strip() for line in raw_header.split("\r\n") if line != ""]
    if header_lines[0] != "######## Neuralynx Data File Header":
        warnings.warn("Unexpected start to header: " + header_lines[0])

    # Try to read the original file path
    try:
        for line in header_lines:
            if "OriginalFileName" in line:
                header["FileName"] = line.split('"')[1]
            if "TimeClosed" in line:
                header["TimeClosed"] = line
                header["TimeClosed_dt"] = parse_neuralynx_time_string(line)
            if "TimeOpened" in line:
                header["TimeOpened"] = line
                header["TimeOpened_dt"] = parse_neuralynx_time_string(line)
    except:
        warnings.warn("Unable to parse info from Neuralynx header")

    # Process lines with file opening and closing times

    # Read the parameters, assuming "-PARAM_NAME PARAM_VALUE" format
    for line in header_lines[4:]:
        try:
            name, value = line[1:].split(
                " ", 1
            )  # Ignore the dash and split PARAM_NAME and PARAM_VALUE
            header[name] = value
        except:
            warnings.warn(
                "Unable to parse parameter line from Neuralynx header: " + line
            )

    return header


def read_records(fid, record_dtype, record_skip=0, count=None):
    # Read count records (default all) from the file object fid skipping the first record_skip records. Restores the
    # position of the file object after reading.
    if count is None:
        count = -1

    pos = fid.tell()
    fid.seek(HEADER_LENGTH, 0)
    fid.seek(record_skip * record_dtype.itemsize, 1)
    rec = np.fromfile(fid, record_dtype, count=count)
    fid.seek(pos)

    return rec


def estimate_record_count(file_path, record_dtype):
    # Estimate the number of records from the file size
    file_size = os.path.getsize(file_path)
    file_size -= HEADER_LENGTH

    if file_size % record_dtype.itemsize != 0:
        warnings.warn(
            "File size is not divisible by record size (some bytes unaccounted for)"
        )

    return file_size / record_dtype.itemsize


def parse_neuralynx_time_string(time_string):
    # Parse a datetime object from the idiosyncratic time string in Neuralynx file headers
    try:
        date = [int(x) for x in time_string.split()[1].split("/")]
        time = [int(x) for x in time_string.split()[-1].replace(".", ":").split(":")]
    except:
        warnings.warn(
            "Unable to parse time string from Neuralynx header: " + time_string
        )
        return None
    else:
        return datetime.datetime(date[0], date[1], date[2], time[0], time[1], time[2])


def check_ncs_records(records):
    # Check that all the records in the array are "similar" (have the same sampling frequency etc.
    dt = np.rint(np.diff(records["TimeStamp"]))
    dt = np.abs(dt - dt[0])
    is_valid = True
    if not np.all(records["ChannelNumber"] == records[0]["ChannelNumber"]):
        warnings.warn("Channel number changed during record sequence")
        is_valid = False
    elif not np.all(records["SampleFreq"] == records[0]["SampleFreq"]):
        warnings.warn("Sampling frequency changed during record sequence")
        is_valid = False
    elif np.any(records["NumValidSamples"] != 512):
        warnings.warn("Invalid samples in one or more records")
        is_valid = False
    elif not np.all(dt <= 1):
        warnings.warn("Time stamp difference tolerance exceeded")
        is_valid = False
    else:
        return is_valid


def load_ncs(
    file_path, load_time=True, rescale_data=True, signal_scaling=MICROVOLT_SCALING
):
    # Load the given file as a Neuralynx .ncs continuous acquisition file and extract the contents
    file_path = os.path.abspath(file_path)
    with open(file_path, "rb") as fid:
        raw_header = read_header(fid)
        records = read_records(fid, NCS_RECORD)

    header = parse_header(raw_header)
    check_ncs_records(records)

    # Reshape (and rescale, if requested) the data into a 1D array
    data = records["Samples"].ravel()
    # data = records['Samples'].reshape((NCS_SAMPLES_PER_RECORD * len(records), 1))
    if rescale_data:
        try:
            # ADBitVolts specifies the conversion factor between the ADC counts and volts
            data = data.astype(np.float64) * (
                np.float64(header["ADBitVolts"]) * signal_scaling[0]
            )
        except KeyError:
            warnings.warn(
                "Unable to rescale data, no ADBitVolts value specified in header"
            )
            rescale_data = False

    # Pack the extracted data in a dictionary that is passed out of the function
    ncs = dict()
    ncs["file_path"] = file_path
    ncs["raw_header"] = raw_header
    ncs["header"] = header
    ncs["data"] = data
    ncs["data_units"] = signal_scaling[1] if rescale_data else "ADC counts"
    ncs["sampling_freq"] = records["SampleFreq"][0]
    ncs["channel"] = records["ChannelNumber"][0]
    ncs["timestamp"] = records["TimeStamp"]

    # Calculate the sample time points (if needed)
    if load_time:
        num_samples = data.shape[0]
        times = np.interp(
            np.arange(num_samples), np.arange(0, num_samples, 512), records["TimeStamp"]
        ).astype(np.uint64)
        ncs["time"] = times
        ncs["time_units"] = "µs"

    return ncs


def load_nev(file_path):
    # Load the given file as a Neuralynx .nev event file and extract the contents
    file_path = os.path.abspath(file_path)
    with open(file_path, "rb") as fid:
        raw_header = read_header(fid)
        records = read_records(fid, NEV_RECORD)

    header = parse_header(raw_header)

    # Check for the packet data size, which should be two. DISABLED because these seem to be set to 0 in our files.
    # assert np.all(record['pkt_data_size'] == 2), 'Some packets have invalid data size'

    # Pack the extracted data in a dictionary that is passed out of the function
    nev = dict()
    nev["file_path"] = file_path
    nev["raw_header"] = raw_header
    nev["header"] = header
    nev["records"] = records
    nev["events"] = records[
        ["pkt_id", "TimeStamp", "event_id", "ttl", "Extra", "EventString"]
    ]

    return nev


if __name__ == "__main__":

    #####
    # ttl triggers are saved in:
    # InptX.ncs, where X is the number.
    # For the flanker task, we are using Inpt2 to Inpt4.
    # Input 2, one tick for every trial
    # Input 3 for incon + congruent trials
    # Input 4 congruent (I know, its messy...)
    # for patient 525: Inpt2
    # for patient 532: Inpt3

    # channel LFP recordings will be in
    # LFPxN.ncs, N is the channel number. So we would have to figure out which channel we want to analyze.
    # for patient 525:
    # for patient 532:

    # the sampling rate for LFP is 2k hz, and for the trigger channel is 16k hz.
    # it appears that there is a data field 'time' that would allow the two to be matached.
    # Will have to write code to match samples.

    # Load triggers
    os.chdir(
        "/mnt/cifs/rdss/rdss_kahwang/ECoG_data/532-040_FlankerTask/2020-09-15_13-11-01"
    )
    raw_dir = (
        "/mnt/cifs/rdss/rdss_kahwang/ECoG_data/532-040_FlankerTask/2020-09-15_13-11-01"
    )
    behavorial_dir = "/mnt/cifs/rdss/rdss_kahwang/ECoG_data/532/"

    trig2 = load_ncs("Inpt2.ncs")
    trig3 = load_ncs("Inpt3.ncs")
    # example plotting
    # sns.lineplot(data=trig3["data"][0:])
    # plt.show()

    # raw trigger channels denoting stimulus - Inpt3 had all trials, other subject had Inpt2 for all trials
    # should check this for each subject
    info = mne.create_info(ch_names=["Inpt2", "Inpt3"], sfreq=trig2["sampling_freq"])
    raw_trig = mne.io.RawArray(
        np.array(
            [
                np.where(trig2["data"] < 1000, 0, trig2["data"]),
                np.where(trig3["data"] < 1000, 0, trig3["data"]),
            ]
        ),
        info,
    )

    # find all events, this subject had 10 extra events in middle
    events = mne.find_events(
        raw_trig, stim_channel="Inpt3", min_duration=(10 / raw_trig.info["sfreq"])
    )
    events = np.delete(events, np.arange(100, 110), axis=0)

    # get behavioral csv files
    behavorial_files = sorted(glob.glob(behavorial_dir + "*.csv"))
    behavorial_df = pd.DataFrame()
    for file in behavorial_files:
        behavorial_df = behavorial_df.append(pd.read_csv(file))

    # set event type to 1 or 2 for congruent and incongruent trials
    for event_index in np.arange(events.shape[0]):
        trial_type = behavorial_df.iloc[event_index]["trial_type"]
        if trial_type == "con":
            events[event_index, 2] = 1
        elif trial_type == "incon":
            events[event_index, 2] = 2
        else:
            raise ValueError("trial type must be con or incon")

    # create mevent annotations
    mapping = {1: "con", 2: "incon"}
    annot_from_events = mne.annotations_from_events(
        events=events,
        event_desc=mapping,
        sfreq=raw_trig.info["sfreq"],
    )

    # load lfps
    lfp_list = glob.glob("LFP*.ncs")
    for index in range(len(lfp_list)):
        lfp_list[index] = load_ncs(lfp_list[index])
    lfp_list.sort(key=lambda x: x["channel"])
    channels = [str(file["channel"]) for file in lfp_list]

    # ensure sampling freq is the same for all lfps
    sampling_freq = lfp_list[0]["sampling_freq"]
    if any(file["sampling_freq"] != sampling_freq for file in lfp_list):
        warnings.warn("Sample frequency does not match for all LFP files.")

    # create raw object
    info = mne.create_info(channels, sfreq=sampling_freq)
    raw_data = [lfp["data"] for lfp in lfp_list]
    raw_channels = mne.io.RawArray(
        np.array(raw_data),
        info,
    )
    raw_channels.set_annotations(annot_from_events)

    raw_channels.plot(start=5, duration=5)
    plt.show()