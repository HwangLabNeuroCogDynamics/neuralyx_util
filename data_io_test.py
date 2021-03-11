# script to test how to read neuralyx data
# modififed from code kindly provided by
# alafuzof (https://github.com/alafuzof)
#https://github.com/alafuzof/NeuralynxIO

import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import numpy as np
import datetime
import glob as glob
import mne


HEADER_LENGTH = 16 * 1024  # 16 kilobytes of header

NCS_SAMPLES_PER_RECORD = 512
NCS_RECORD = np.dtype([('TimeStamp',       np.uint64),       # Cheetah timestamp for this record. This corresponds to
                                                             # the sample time for the first data point in the Samples
                                                             # array. This value is in microseconds.
                       ('ChannelNumber',   np.uint32),       # The channel number for this record. This is NOT the A/D
                                                             # channel number
                       ('SampleFreq',      np.uint32),       # The sampling frequency (Hz) for the data stored in the
                                                             # Samples Field in this record
                       ('NumValidSamples', np.uint32),       # Number of values in Samples containing valid data
                       ('Samples',         np.int16, NCS_SAMPLES_PER_RECORD)])  # Data points for this record. Cheetah
                                                                                # currently supports 512 data points per
                                                                                # record. At this time, the Samples
                                                                                # array is a [512] array.

NEV_RECORD = np.dtype([('stx',           np.int16),      # Reserved
                       ('pkt_id',        np.int16),      # ID for the originating system of this packet
                       ('pkt_data_size', np.int16),      # This value should always be two (2)
                       ('TimeStamp',     np.uint64),     # Cheetah timestamp for this record. This value is in
                                                         # microseconds.
                       ('event_id',      np.int16),      # ID value for this event
                       ('ttl',           np.int16),      # Decimal TTL value read from the TTL input port
                       ('crc',           np.int16),      # Record CRC check from Cheetah. Not used in consumer
                                                         # applications.
                       ('dummy1',        np.int16),      # Reserved
                       ('dummy2',        np.int16),      # Reserved
                       ('Extra',         np.int32, 8),   # Extra bit values for this event. This array has a fixed
                                                         # length of eight (8)
                       ('EventString',   'S', 128)])  # Event string associated with this event record. This string
                                                         # consists of 127 characters plus the required null termination
                                                         # character. If the string is less than 127 characters, the
                                                         # remainder of the characters will be null.

VOLT_SCALING = (1, u'V')
MILLIVOLT_SCALING = (1000, u'mV')
MICROVOLT_SCALING = (1000000, u'µV')


def read_header(fid):
    # Read the raw header data (16 kb) from the file object fid. Restores the position in the file object after reading.
    pos = fid.tell()
    fid.seek(0)
    raw_header = fid.read(HEADER_LENGTH).strip(b'\0')
    fid.seek(pos)

    return raw_header


def parse_header(raw_header):
    # Parse the header string into a dictionary of name value pairs
    header = dict()

    # Decode the header as iso-8859-1 (the spec says ASCII, but there is at least one case of 0xB5 in some headers)
    raw_header = raw_header.decode('iso-8859-1')

    # Neuralynx headers seem to start with a line identifying the file, so
    # let's check for it
    header_lines = [line.strip() for line in raw_header.split('\r\n') if line != '']
    if header_lines[0] != '######## Neuralynx Data File Header':
        warnings.warn('Unexpected start to header: ' + header_lines[0])

    # Try to read the original file path
    try:
        for line in header_lines:
            if 'OriginalFileName' in line:
                header[u'FileName']  = line.split('\"')[1]
            if 'TimeClosed' in line:
                header[u'TimeClosed'] = line
                header[u'TimeClosed_dt'] = parse_neuralynx_time_string(line)
            if 'TimeOpened' in line:
                header[u'TimeOpened'] = line
                header[u'TimeOpened_dt'] = parse_neuralynx_time_string(line)
    except:
        warnings.warn('Unable to parse info from Neuralynx header')

    # Process lines with file opening and closing times


    # Read the parameters, assuming "-PARAM_NAME PARAM_VALUE" format
    for line in header_lines[4:]:
        try:
            name, value = line[1:].split(' ', 1)  # Ignore the dash and split PARAM_NAME and PARAM_VALUE
            header[name] = value
        except:
            warnings.warn('Unable to parse parameter line from Neuralynx header: ' + line)

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
        warnings.warn('File size is not divisible by record size (some bytes unaccounted for)')

    return file_size / record_dtype.itemsize


def parse_neuralynx_time_string(time_string):
    # Parse a datetime object from the idiosyncratic time string in Neuralynx file headers
    try:
        date = [int(x) for x in time_string.split()[1].split('/')]
        time = [int(x) for x in time_string.split()[-1].replace('.', ':').split(':')]
    except:
        warnings.warn('Unable to parse time string from Neuralynx header: ' + time_string)
        return None
    else:
        return datetime.datetime(date[0], date[1], date[2],
                                 time[0], time[1], time[2])

def check_ncs_records(records):
    # Check that all the records in the array are "similar" (have the same sampling frequency etc.
    dt = np.rint(np.diff(records['TimeStamp']))
    dt = np.abs(dt - dt[0])
    is_valid = True
    if not np.all(records['ChannelNumber'] == records[0]['ChannelNumber']):
        warnings.warn('Channel number changed during record sequence')
        is_valid = False
    elif not np.all(records['SampleFreq'] == records[0]['SampleFreq']):
        warnings.warn('Sampling frequency changed during record sequence')
        is_valid = False
    elif np.any(records['NumValidSamples'] != 512):
        warnings.warn('Invalid samples in one or more records')
        is_valid = False
    elif not np.all(dt <= 1):
        warnings.warn('Time stamp difference tolerance exceeded')
        is_valid = False
    else:
        return is_valid


def load_ncs(file_path, load_time=True, rescale_data=True, signal_scaling=MICROVOLT_SCALING):
    # Load the given file as a Neuralynx .ncs continuous acquisition file and extract the contents
    file_path = os.path.abspath(file_path)
    with open(file_path, 'rb') as fid:
        raw_header = read_header(fid)
        records = read_records(fid, NCS_RECORD)

    header = parse_header(raw_header)
    check_ncs_records(records)

    # Reshape (and rescale, if requested) the data into a 1D array
    data = records['Samples'].ravel()
    #data = records['Samples'].reshape((NCS_SAMPLES_PER_RECORD * len(records), 1))
    if rescale_data:
        try:
            # ADBitVolts specifies the conversion factor between the ADC counts and volts
            data = data.astype(np.float64) * (np.float64(header['ADBitVolts']) * signal_scaling[0])
        except KeyError:
            warnings.warn('Unable to rescale data, no ADBitVolts value specified in header')
            rescale_data = False

    # Pack the extracted data in a dictionary that is passed out of the function
    ncs = dict()
    ncs['file_path'] = file_path
    ncs['raw_header'] = raw_header
    ncs['header'] = header
    ncs['data'] = data
    ncs['data_units'] = signal_scaling[1] if rescale_data else 'ADC counts'
    ncs['sampling_freq'] = records['SampleFreq'][0]
    ncs['channel'] = records['ChannelNumber'][0]
    ncs['timestamp'] = records['TimeStamp']

    # Calculate the sample time points (if needed)
    if load_time:
        num_samples = data.shape[0]
        times = np.interp(np.arange(num_samples), np.arange(0, num_samples, 512), records['TimeStamp']).astype(np.uint64)
        ncs['time'] = times
        ncs['time_units'] = u'µs'

    return ncs


def load_nev(file_path):
    # Load the given file as a Neuralynx .nev event file and extract the contents
    file_path = os.path.abspath(file_path)
    with open(file_path, 'rb') as fid:
        raw_header = read_header(fid)
        records = read_records(fid, NEV_RECORD)

    header = parse_header(raw_header)

    # Check for the packet data size, which should be two. DISABLED because these seem to be set to 0 in our files.
    #assert np.all(record['pkt_data_size'] == 2), 'Some packets have invalid data size'


    # Pack the extracted data in a dictionary that is passed out of the function
    nev = dict()
    nev['file_path'] = file_path
    nev['raw_header'] = raw_header
    nev['header'] = header
    nev['records'] = records
    nev['events'] = records[['pkt_id', 'TimeStamp', 'event_id', 'ttl', 'Extra', 'EventString']]

    return nev



if __name__ == "__main__":

#####
# ttl triggers are saved in:
# InptX.ncs, where X is the number.
# For the flanker task, we are using Inpt2 to Inpt4.
# Input 2, one tick for every trial
# Input 3 for incon + congruent trials
# Input 4 congruent (I know, its messy...)

# channel LFP recordings will be in
# LFPxN.ncs, N is the channel number. So we would have to figure out which channel we want to analyze.
# for patient 525:
# for patient 532:

# the sampling rate for LFP is 2k hz, and for the trigger channel is 16k hz.
# it appears that there is a data field 'time' that would allow the two to be matached.
# Will have to write code to match samples.

# Load triggers
    os.chdir('/mnt/cifs/rdss/rdss_kahwang/ECoG_data/525-040_Flanker/2020-08-19_16-01-34')

    trig1 = load_ncs('Inpt1.ncs')
    trig2 = load_ncs('Inpt2.ncs')
    trig3 = load_ncs('Inpt3.ncs')

    # example plotting
    # sns.lineplot(data=trig1['data'][0:])
    # sns.lineplot(data=trig2['data'][0:])
    # sns.lineplot(data=trig3['data'][0:])
    # plt.show()
    # load LFP
    # sns.lineplot(data=lfp['data'][1250000:1300000])
    # plt.show()
    print(trig2['data'])
    arr1 = np.where(trig1['data'] > 1000)[0]
    arr2 = np.where(trig2['data'] > 1000)[0]
    arr3 = np.where(trig3['data'] > 1000)[0]
    
    is_consecutive = True
    epochs = []
    current_epoch = []
    for index in np.arange(arr2.size):
        if index != 0 and abs(arr2[index] - arr2[index - 1]) > 50:
            epochs.append(current_epoch)
            current_epoch = []
        if index == arr2.size - 1:
            current_epoch.append(arr2[index])
            epochs.append(current_epoch)
        current_epoch.append(arr2[index])
        
        lfp_list = glob.glob('LFP*.ncs')
    for index in range(len(lfp_list)):
        lfp_list[index] = load_ncs(lfp_list[index])
    lfp_list.sort(key=lambda x: x['channel'])
    channels = [str(file['channel']) for file in lfp_list]

    sampling_freq = lfp_list[0]['sampling_freq']
    length = lfp_list[0]['time'][-1] - lfp_list[0]['time'][0]
    freq = length / len(lfp_list[0]['time'])
    if any(file['sampling_freq'] != sampling_freq for file in lfp_list):
        warnings.warn('Sample frequency does not match for all LFP files.')
            
    info = mne.create_info(channels, sfreq=sampling_freq)
    raw_data = [lfp['data'] for lfp in lfp_list]

    events = np.column_stack((np.arange(0, length, sampling_freq),
                        np.zeros(200, dtype=int),
                        np.array([1, 2, 1, 2, 1])))
    event_dict = dict(congruent=1, incongruent=2)
    simulated_epochs = mne.EpochsArray(raw_data, info, tmin=-0.5, events=events,
                                    event_id=event_dict)
    simulated_epochs.plot(picks='misc', show_scrollbars=False, events=events,
                        event_id=event_dict)

    plt.show()
    