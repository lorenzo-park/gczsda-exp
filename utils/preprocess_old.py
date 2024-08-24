import pywt
import warnings
warnings.filterwarnings("ignore")

from unpack.unpackfloat import unpack_float
from scapy.all import *
#from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy import signal

import numpy as np
import math

BW = 80

HOFFSET = 16
NFFT = int(BW * 3.2)

#for out-of-band signal filtering
#uses hard-coded mac addresses to detect Wi-Fi channels and create mask
#TODO: auto-detect channel with chanspec instead of hard-coding it?????
mac_list_ch36 = [b'\xbe\xbev\x89a\x94',b'\xa2\xbev\x89a\x94',b'\xa6\xbev\x89a\x94',b'\xaa\xbev\x89a\x94',b'\xb0\xbev\x89a\x94',b'\xb2\xbev\x89a\x94',b'\xb6\xbev\x89a\x94',b'\xba\xbev\x89a\x94',b'.\x80\x88\xec*a',b'"\x80\x88\xec*a',b'&\x80\x88\xec*a',b':\x80\x88\xec*a',b'>\x80\x88\xec*a',b'2\x80\x88\xec*a',b'(\x80\x88\xec*a',b'*\x80\x88\xec*a',b'\xf6\x08kd\x9f\xb1',b'\xec\x08kd\x9f\xb1',b'\xee\x08kd\x9f\xb1',b'\xea\x08kd\x9f\xb1',b'\xe6\x08kd\x9f\xb1',b'\xe2\x08kd\x9f\xb1',b'\xfe\x08kd\x9f\xb1',b'\xfa\x08kd\x9f\xb1',b'&\x80\x88\xec,\xad', b':\x80\x88\xec,\xad', b'>\x80\x88\xec,\xad', b'2\x80\x88\xec,\xad', b'(\x80\x88\xec,\xad', b'*\x80\x88\xec,\xad', b'.\x80\x88\xec,\xad', b'"\x80\x88\xec,\xad']
mac_list_ch40 = [b'\xe6\x08kd\x9f\xae',b'\xe2\x08kd\x9f\xae',b'\xfe\x08kd\x9f\xae',b'\xfa\x08kd\x9f\xae',b'\xf6\x08kd\x9f\xae',b'\xec\x08kd\x9f\xae',b'\xee\x08kd\x9f\xae',b'\xea\x08kd\x9f\xae',b'.\x80\x88\xec,\xec',b'"\x80\x88\xec,\xec', b'&\x80\x88\xec,\xec',b':\x80\x88\xec,\xec',b'>\x80\x88\xec,\xec',b'2\x80\x88\xec,\xec',b'(\x80\x88\xec,\xec',b'*\x80\x88\xec,\xec',b'\xcaG2\xb3a\xbf',b'\xceG2\xb3a\xbf',b'\xc2G2\xb3a\xbf',b'\xd8G2\xb3a\xbf',b'\xdeG2\xb3a\xbf',b'\xd2G2\xb3a\xbf',b'\xd6G2\xb3a\xbf',b'\xdaG2\xb3a\xbf', b'>\x80\x88\xec1\x0c', b'2\x80\x88\xec1\x0c',b'(\x80\x88\xec1\x0c',b'*\x80\x88\xec1\x0c',b'.\x80\x88\xec1\x0c',b'&\x80\x88\xec1\x0c', b':\x80\x88\xec1\x0c']
mac_list_ch44 = [b'\xea\x08kb\xe4\x85',b'\xe6\x08kb\xe4\x85', b'\xe2\x08kb\xe4\x85',b'\xfe\x08kb\xe4\x85',b'\xfa\x08kb\xe4\x85',b'\xf6\x08kb\xe4\x85',b'\xec\x08kb\xe4\x85',b'\xee\x08kb\xe4\x85',b'&\x80\x88\xec-\n',b':\x80\x88\xec-\n',b'>\x80\x88\xec-\n',b'2\x80\x88\xec-\n',b'(\x80\x88\xec-\n',b'*\x80\x88\xec-\n',b'.\x80\x88\xec-\n',b'"\x80\x88\xec-\n',b'\xdeG2\xb3_m',b'\xd2G2\xb3_m',b'\xd6G2\xb3_m',b'\xcaG2\xb3_m', b'\xceG2\xb3_m',b'\xd8G2\xb3_m',b'\xdaG2\xb3_m',b'\xc2G2\xb3_m', b'(\x80\x88\xec.i', b'*\x80\x88\xec.i', b'.\x80\x88\xec.i', b'"\x80\x88\xec.i', b'&\x80\x88\xec.i', b':\x80\x88\xec.i', b'>\x80\x88\xec.i', b'2\x80\x88\xec.i']
mac_list_ch48 = [b'>\x80\x88\xec0\xac',b'(\x80\x88\xec0\xac',b'*\x80\x88\xec+\xe4',b'.\x80\x88\xec+\xe4',b'"\x80\x88\xec+\xe4',b'&\x80\x88\xec+\xe4',b':\x80\x88\xec+\xe4',b'>\x80\x88\xec+\xe4',b'2\x80\x88\xec+\xe4',b'(\x80\x88\xec+\xe4',b'2\x80\x88\xec0\xac',b'*\x80\x88\xec0\xac',b'.\x80\x88\xec0\xac',b'"\x80\x88\xec0\xac',b'&\x80\x88\xec0\xac',b':\x80\x88\xec0\xac']
mac_list = mac_list_ch36 + mac_list_ch40 + mac_list_ch44 + mac_list_ch48

mask_36 = np.zeros(256)
mask_36[0+6:64-5] = np.ones(53)
mask_36[32] = 0

mask_40 = np.zeros(256)
mask_40[64+6:128-5] = np.ones(53)
mask_40[96] = 0

mask_44 = np.zeros(256)
mask_44[128+6:192-5] = np.ones(53)
mask_44[160] = 0

mask_48 = np.zeros(256)
mask_48[192+6:256-5] = np.ones(53)
mask_48[224] = 0

spectrum_mask_dict = {}
mac_dict = {}

for mac in mac_list_ch36:
    spectrum_mask_dict[mac] = mask_36
    mac_dict[mac] = 36

for mac in mac_list_ch40:
    spectrum_mask_dict[mac] = mask_40
    mac_dict[mac] = 40

for mac in mac_list_ch44:
    spectrum_mask_dict[mac] = mask_44
    mac_dict[mac] = 44

for mac in mac_list_ch48:
    spectrum_mask_dict[mac] = mask_48
    mac_dict[mac] = 48

def dc_subcarrier_interpolate(spectrogram):
    for i in range(0, len(spectrogram)):
        spectrogram[i][32] = (spectrogram[i][31] + spectrogram[i][33])/2
        spectrogram[i][96] = (spectrogram[i][95] + spectrogram[i][97])/2
        spectrogram[i][160] = (spectrogram[i][159] + spectrogram[i][161])/2
        spectrogram[i][224] = (spectrogram[i][223] + spectrogram[i][225])/2
    return spectrogram

#numerical differentiation of the spectrogram in time axis
#if input dim is [1000, 256] , output dim is [999, 256]
def numerical_differentiation(spectrogram):
    spectrogram_out = np.zeros([len(spectrogram)-1, len(spectrogram[0])]) #1000, 256

    for i in range(0, len(spectrogram)-1):
        spectrogram_out[i] = spectrogram[i+1] - spectrogram[i]

    return spectrogram_out, spectrogram[0]

#numerical integration of the spectrogram in time axis
def numerical_integration(d_spectrogram, c):
    spectrogram_out = np.zeros([len(d_spectrogram)+1, len(d_spectrogram[0])]) #1000, 256
    spectrogram_out[0] = c;

    for i in range(0, len(d_spectrogram)):
        spectrogram_out[i+1] = spectrogram_out[i] + d_spectrogram[i]

    return spectrogram_out

#remove outliers in numerical differentiation results
#(channelwise, time invariant version. Simply nulls out outliers.)
def numerical_differentiation_outlier_removal_v1(spectrogram_d, threshold):
    spectrogram_out = spectrogram_d
    foo = int(len(spectrogram_out[0])/64)

    for i in range(0, len(spectrogram_d)):
        for channel in range (0, foo): #channelwise:
            if(np.abs(np.mean(spectrogram_out[i][channel*64:channel*64+64])) > threshold):
                spectrogram_out[i][channel*64:channel*64+64] = np.zeros(64)
    return spectrogram_out

def numerical_differentiation_outlier_removal_subcarrierwise(spectrogram_d, threshold_sigma = 3):
    spectrogram_out = spectrogram_d    #1000 x 256
    foo = int(len(spectrogram_out[0])/64)
    means = np.mean(spectrogram_d, axis=0)
    stds = np.std(spectrogram_d, axis=0)
    thres = np.add(means, stds*threshold_sigma)

    for i in range(0, len(spectrogram_d[0])):
        spectrogram_out[:, i] = np.where(spectrogram_d[:, i] > thres[i], 0, spectrogram_d[:, i])

    return spectrogram_out

#reorders channels
def reorder_channels(spectrogram, order):
    spectrogram_out = np.zeros([len(spectrogram), 64*len(order)]) #1000, 256

    for i in range(0, len(spectrogram)):
        for channel in range(0, len(order)):
            spectrogram_out[i][channel*64:channel*64+64] = spectrogram[i][order[channel]*64:order[channel]*64+64]

    return spectrogram_out

#Filter packets (keeps packests from whitelisted Wi-Fi STAs only)
def filter_packets(packets, whitelist, verbose=False):
    packets_out = scapy.plist.PacketList()
    drop_count = 0
    drop_mac_list=[]
    for i in range (0, len(packets)):
        payload = bytes(packets[i][UDP].payload)
        mac = bytes(np.frombuffer(payload, dtype=np.uint8)[4:10])
        if mac in whitelist:
            packets_out.append(packets[i])
        else:
            drop_count= drop_count + 1
            if(verbose and mac not in drop_mac_list):
                print("dropped:" + format(mac[0],'x')+":"+format(mac[1],'x')+":"+format(mac[2],'x')+":"+format(mac[3],'x')+":"+format(mac[4],'x')+":"+format(mac[5],'x') + "(" + str(mac) + ")")
                drop_mac_list.append(mac)


    if(verbose and drop_count >=1):
        print("total dropped:" + str(drop_count))
    return packets_out


#Gets MAC address from the NexMon packet
def get_mac_list(packets):
    mac_dict = {}
    for packet in packets:
        payload = bytes(packet[UDP].payload)
        mac = bytes(np.frombuffer(payload, dtype=np.uint8)[4:10])
        if mac not in mac_dict:
            mac_dict[mac] = 1
        else:
            mac_dict[mac] = mac_dict[mac] + 1

    return mac_dict

def CSI_Interpolation(threshold_factor, amplitudes_mat):
    spectrogram = amplitudes_mat

    #CSI Interpolation
    psd = np.mean(spectrogram, axis=0)
    mean_level_per_channel  = np.zeros(4)
    threshold = np.zeros(4)

    for i in range (0, 4):
        for j in range (0, 64):
            mean_level_per_channel [i] += psd[i*64 + j]
        threshold[i] = mean_level_per_channel [i] / threshold_factor;

    #forward
    for i in range (1, len(spectrogram)):
        tmp = spectrogram[i]
        for j in range (0, 4):
            if(sum(tmp[j*64:j*64+64]) < threshold[j]):
                spectrogram[i][j*64:j*64+64] = spectrogram[i-1][j*64:j*64+64]

    #backward
    for i in range (int(len(spectrogram)/4), -1, -1):
        tmp = spectrogram[i]
        for j in range (0, 4):
            if(sum(tmp[j*64:j*64+64]) < threshold[j]):
                spectrogram[i][j*64:j*64+64] = spectrogram[i+1][j*64:j*64+64]

    return spectrogram


def AGC_cancellation(spectrogram, cluster_size = 5, normalized_amplitude = 10, do_skip_outliers=False, outlier_min = 0, outlier_max = 3):
    amplitude_per_channel = np.zeros([4,len(spectrogram)])
    for i in range (0, len(spectrogram)):
        amplitude_per_channel[0][i] = np.mean(spectrogram[i][0:64])
        amplitude_per_channel[1][i] = np.mean(spectrogram[i][64:128])
        amplitude_per_channel[2][i] = np.mean(spectrogram[i][128:192])
        amplitude_per_channel[3][i] = np.mean(spectrogram[i][192:256])

    for channel in range(0, 4):
        kmeans = KMeans(n_clusters=cluster_size, random_state=1337).fit(amplitude_per_channel[channel].reshape(-1,1))

        #kmeans = AgglomerativeClustering(linkage='ward', n_clusters=None, distance_threshold=np.mean(amplitude_per_channel[channel])/5).fit(amplitude_per_channel[channel].reshape(-1,1)) #TODO improve this

        for i in range(0, cluster_size):
            positions = np.where(kmeans.labels_ == i)
            avg_amplitude = np.mean(amplitude_per_channel[channel][positions])
            correction_factor = 10.0/(avg_amplitude+1e-50)
            if do_skip_outliers and (correction_factor < outlier_min or correction_factor > outlier_max):
                continue

            for position in positions[0]:
                spectrogram[position][channel*64:channel*64+64] = spectrogram[position][channel*64:channel*64+64] * correction_factor

    return spectrogram

def mono_to_color(
    X: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6
):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

#TODO fix (very, very poor implementation...)
def not_so_mono_to_color(
    X1: np.ndarray,X2: np.ndarray,X3: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6
):
    V1 = mono_to_color(X1, mean, std, norm_max, norm_min, eps)
    V2 = mono_to_color(X2, mean, std, norm_max, norm_min, eps)
    V3 = mono_to_color(X3, mean, std, norm_max, norm_min, eps)

    V1 = V1.T
    V1[1] = V2.T[0]
    V1[2] = V3.T[0]
    V1 = V1.T

    return V1

def parse_packet(packet):
    rx_time = packet.time
    payload = bytes(packet[UDP].payload)
    rssi = np.frombuffer(payload, dtype=np.int8)[3]
    mac = bytes(np.frombuffer(payload, dtype=np.uint8)[4:10])
    H_uint8 = bytes(np.frombuffer(payload, dtype=np.uint8)[HOFFSET+2:HOFFSET+2+NFFT*4])
    frame_seq_no = np.frombuffer(payload, dtype=np.uint16)[5]
    stream_core_no = np.frombuffer(payload, dtype=np.uint16)[6]
    chanspec = np.frombuffer(payload, dtype=np.uint16)[7]
    chip_ver = np.frombuffer(payload, dtype=np.uint16)[8]

    return payload, mac, H_uint8, rx_time, frame_seq_no, stream_core_no, chanspec, chip_ver, rssi


def IQ_to_polar_coordinate(packet_iqs, is_degree=True):
    packet_iqs_mat = np.concatenate([packet_iqs], axis=0)
    packets_i = np.fft.fftshift(packet_iqs_mat[:, :, 0], axes=(1,))
    packets_q = np.fft.fftshift(packet_iqs_mat[:, :, 1], axes=(1,))
    amplitudes = []
    angles = []
    for i, q in zip(packets_i, packets_q):
        z = np.zeros(i.shape, dtype=complex)
        z.real = i * 1e-3
        z.imag = q * 1e-3
        amplitude = np.abs(z)
        amplitudes.append(amplitude)
        angle = np.angle(z, deg=is_degree)
        angles.append(angle)
    amplitudes_mat = np.asarray(amplitudes)
    angles_mat = np.asarray(angles)
    return amplitudes_mat, angles_mat





def get_csi_from_packets(packets, antenna_no = 0, filter_oob = False):
    packet_iqs = []
    macs = []
    rssi_list = []
    for packet in packets:
        payload, mac, H_uint8, rx_time, frame_seq_no, stream_core_no, chanspec, chip_ver, rssi = parse_packet(packet)
        if(antenna_no == stream_core_no):
            H_uint32 = np.frombuffer(H_uint8, dtype=np.uint32)
            Hout = unpack_float(10, 1, 0, 1, 9, 5, NFFT, H_uint32)
            packet_iqs.append(np.reshape(Hout, [len(H_uint32), 2]))
            macs.append(mac)
            rssi_list.append(rssi)

    channel_list = []
    amplitudes_mat, angles_mat = IQ_to_polar_coordinate(packet_iqs)

    for i in range (0, len(amplitudes_mat)):
        if macs[i] in spectrum_mask_dict:
            channel_list.append(mac_dict[macs[i]])
        else:
            channel_list.append(-1)

    if(filter_oob):
        for i in range (0, len(amplitudes_mat)):
            if macs[i] in spectrum_mask_dict:
                mask = spectrum_mask_dict[macs[i]]
            else:
                mask = np.ones(256)
                print("warning: out-of-band filtering failure (no mask)")
            amplitudes_mat[i] = np.multiply(amplitudes_mat[i], mask)
            angles_mat[i] = np.multiply(angles_mat[i], mask)

    return amplitudes_mat, angles_mat, channel_list, rssi_list


def get_csi_from_packets_mimo (packets, filter_oob = False, ch_no = -1):
    packet_iqs = []
    stream_nos = []
    macs = []
    macs_groupped = []
    rssi_list = []
    frame_seq_nos = []
    timestamps = []
    timestamp_origin = -1
    _mac_list = mac_list
    if(ch_no == 36):
        _mac_list = mac_list_ch36
    elif(ch_no == 40):
        _mac_list = mac_list_ch40
    elif(ch_no == 44):
        _mac_list = mac_list_ch44
    elif(ch_no == 48):
        _mac_list = mac_list_ch48

    #for each packet in pcap:
    for packet in packets:

        #get data
        payload, mac, H_uint8, rx_time, frame_seq_no, stream_core_no, chanspec, chip_ver, rssi = parse_packet(packet)
        if mac in _mac_list:
            H_uint32 = np.frombuffer(H_uint8, dtype=np.uint32)
            Hout = unpack_float(10, 1, 0, 1, 9, 5, NFFT, H_uint32)
            packet_iqs.append(np.reshape(Hout, [len(H_uint32), 2]))
            stream_nos.append(stream_core_no)
            macs.append(mac)
            frame_seq_nos.append(frame_seq_no)
            rssi_list.append(rssi)
            timestamps.append(rx_time)
            if(timestamp_origin == -1):
                timestamp_origin = rx_time
            #print(f'mac:{mac}, frame: {frame_seq_no}, stream_core_no:{stream_core_no}, chansepc:{chanspec}')

    if(len(rssi_list) == 0):
        return None, None, None, None, None

    #re-format to np.complex
    packet_iqs_mat = np.concatenate([packet_iqs], axis=0)

    packet_iqs_reformatted = packet_iqs_mat.astype(np.float32).view(dtype=np.complex64)[:,:,0] * 1e-3
    packet_iqs_reformatted = np.fft.fftshift(packet_iqs_reformatted, axes=(1,))

    packet_iqs_ch1 = []
    packet_iqs_ch2 = []
    rssi_list_filtered = []
    timestamps_filtered = []

    for i in range(0, len(packet_iqs_reformatted)-1):
        if(macs[i] == macs[i+1] and frame_seq_nos[i] == frame_seq_nos[i+1] and stream_nos[i] != stream_nos[i+1]):
            macs_groupped.append(macs[i])
            timestamps_filtered.append(timestamps[i] - timestamp_origin)

            mask = spectrum_mask_dict[macs[i]]
            packet_iqs_reformatted[i] = np.multiply(packet_iqs_reformatted[i], mask)
            packet_iqs_reformatted[i+1] = np.multiply(packet_iqs_reformatted[i+1], mask)

            if(stream_nos[i] == 0):
                packet_iqs_ch1.append(packet_iqs_reformatted[i])
                packet_iqs_ch2.append(packet_iqs_reformatted[i+1])
                rssi_list_filtered.append(rssi_list[i])

            else:
                packet_iqs_ch1.append(packet_iqs_reformatted[i+1])
                packet_iqs_ch2.append(packet_iqs_reformatted[i])
                rssi_list_filtered.append(rssi_list[i+1])

    if(len(rssi_list_filtered) == 0):
        return None, None, None, None, None

    return packet_iqs_ch1, packet_iqs_ch2, rssi_list_filtered, macs_groupped, timestamps_filtered


def get_csi_from_packets_aoa (packets, ch_no, filter_oob = False ):

    cal = 0

    if(ch_no == 36):
        mac_list = mac_list_ch36
        cal = 4.877269305239476
    elif(ch_no == 40):
        mac_list = mac_list_ch40
        cal = 4.582965388642128
    elif(ch_no == 44):
        mac_list = mac_list_ch44
        cal = 4.430995725245726
    elif(ch_no == 48):
        mac_list = mac_list_ch48
        cal = 4.20
    else:
        raise Exception("WTF")


    #separate Rx channels
    packet_iqs_ch1, packet_iqs_ch2, rssi_list_ch1, _, _  = get_csi_from_packets_mimo (packets, filter_oob, ch_no)

    if(rssi_list_ch1 is None):
        return None, None, None, None, None

    ch1_LoS = []
    ch2_LoS = []
    LoS_phase_diff = []
    argmax_val = []

    #ifft, max
    for i in range (0, len(packet_iqs_ch1)):
        packet_iqs_ch1[i] = np.fft.ifft(np.fft.ifftshift(packet_iqs_ch1[i]))
        packet_iqs_ch2[i] = np.fft.ifft(np.fft.ifftshift(packet_iqs_ch2[i]))

        #trim
        #packet_iqs_ch1[i] = packet_iqs_ch1[i][0:40]
        #packet_iqs_ch2[i] = packet_iqs_ch2[i][0:40]

        #ch1 = packet_iqs_ch1[i][np.argmax(np.abs(packet_iqs_ch1[i]))]
        #ch2 = packet_iqs_ch2[i][np.argmax(np.abs(packet_iqs_ch2[i]))]

        argmax = np.argmax(np.abs(packet_iqs_ch1[i])**2+np.abs(packet_iqs_ch2[i])**2)
        argmax_val.append(argmax)
        ch1 = packet_iqs_ch1[i][argmax]
        ch2 = packet_iqs_ch2[i][argmax]

        ch1_LoS.append(ch1)
        ch2_LoS.append(ch2)
        phase_diff = np.angle(ch1, deg = False) - np.angle(ch2, deg = False)


        if(phase_diff < 0):   #Not sure why I need this, but it fixes the problem
            phase_diff = (math.pi * 2) + phase_diff
        LoS_phase_diff.append(phase_diff)

    return packet_iqs_ch1, packet_iqs_ch2, [x - cal for x in LoS_phase_diff], argmax_val, rssi_list_ch1

def get_csi_from_packets_gesture_recognition (packets, filter_oob = False, agc_correction=True, threshold_sigma = 3):

    packet_iqs_ch1, packet_iqs_ch2, rssi_list_filtered, macs_groupped, timestamps = get_csi_from_packets_mimo (packets, filter_oob)
    if(packet_iqs_ch1 is None):
        return None, None, None


    #Get amplitudes and angles
    amplitudes_mat_ch1 = np.abs(packet_iqs_ch1)
    amplitudes_mat_ch2 = np.abs(packet_iqs_ch2)
    angles_mat_ch1 = np.angle(packet_iqs_ch1)
    angles_mat_ch2 = np.angle(packet_iqs_ch2)


    #amplitudes_mat_ch1 = dc_subcarrier_interpolate(amplitudes_mat_ch1)
    #amplitudes_mat_ch2 = dc_subcarrier_interpolate(amplitudes_mat_ch2)

    #mask noise
    for i in range (0, len(angles_mat_ch1)):
        if macs_groupped[i] in spectrum_mask_dict:
            mask = spectrum_mask_dict[macs_groupped[i]]
        else:
            print("Missing Mask!!!")

        angles_mat_ch1[i] = np.multiply(angles_mat_ch1[i], mask)
        angles_mat_ch2[i] = np.multiply(angles_mat_ch2[i], mask)

        if(filter_oob):
            amplitudes_mat_ch1[i] = np.multiply(amplitudes_mat_ch1[i], mask)
            amplitudes_mat_ch2[i] = np.multiply(amplitudes_mat_ch2[i], mask)

    angles_diff = np.subtract(angles_mat_ch1,angles_mat_ch2)
    angles_diff = CSI_angle_Interpolation(angles_diff)


    #phase wrap fix
    angles_diff_tmp = angles_diff.copy()
    np.add(math.pi *2, angles_diff, out = angles_diff, where= angles_diff_tmp < 0)

    angles_diff = np.unwrap(angles_diff, axis=0)
    #angles_diff = dc_subcarrier_interpolate(angles_diff)

    for i in range (1, len(angles_diff)):
        for j in range (0, len(angles_diff[0])):
            if((angles_diff[i-1][j] - angles_diff[i][j]) > 2.5):   #Not sure why I need this, but it fixes the problem
                angles_diff[i][j] = math.pi + angles_diff[i][j]
            elif((angles_diff[i-1][j] - angles_diff[i][j]) < -2.5):   #Not sure why I need this, but it fixes the problem
                angles_diff[i][j] = angles_diff[i][j] - math.pi


    #AGC fix
    amplitudes_mat_ch1 = CSI_Interpolation(3,amplitudes_mat_ch1)
    amplitudes_mat_ch2 = CSI_Interpolation(3,amplitudes_mat_ch2)
    if(agc_correction):
        amplitudes_mat_ch1, c = numerical_differentiation(amplitudes_mat_ch1)
        amplitudes_mat_ch1 = numerical_differentiation_outlier_removal_v1(amplitudes_mat_ch1,0.16)
        amplitudes_mat_ch1 = numerical_integration(amplitudes_mat_ch1, c)
        amplitudes_mat_ch2, c = numerical_differentiation(amplitudes_mat_ch2)
        amplitudes_mat_ch2 = numerical_differentiation_outlier_removal_v1(amplitudes_mat_ch2,0.16)
        amplitudes_mat_ch2 = numerical_integration(amplitudes_mat_ch2, c)

    angles_diff, c = numerical_differentiation(angles_diff)
    angles_diff = numerical_differentiation_outlier_removal_subcarrierwise(angles_diff, threshold_sigma = threshold_sigma)
    angles_diff = numerical_integration(angles_diff, c)

    return amplitudes_mat_ch1, amplitudes_mat_ch2, angles_diff, rssi_list_filtered, timestamps

def CSI_angle_Interpolation(angles_mat):
    angles_interpolated = angles_mat

    #forward
    for i in range (1, len(angles_interpolated)):
        tmp = np.abs(angles_interpolated[i])
        for j in range (0, 4):
            if(sum(tmp[j*64:j*64+64]) < 1.0e-5):
                angles_interpolated[i][j*64:j*64+64] = angles_interpolated[i-1][j*64:j*64+64]

    #backward
    for i in range (len(angles_interpolated)-2, -1, -1):
        tmp = np.abs(angles_interpolated[i])
        for j in range (0, 4):
            if(sum(tmp[j*64:j*64+64]) < 1.0e-5):
                angles_interpolated[i][j*64:j*64+64] = angles_interpolated[i+1][j*64:j*64+64]

    return angles_interpolated

#rolling statistics
#copied from: https://stackoverflow.com/questions/27427618/how-can-i-simply-calculate-the-rolling-moving-variance-of-a-time-series-in-pytho/47878011
#by Josh Albert
def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

#todo: add filter
#todo: add time
#todo: add 64/128pt fft
def channelwise_activity_measure(angles_diff, baz = 64):#todo: add filter
    moving_vars = np.zeros([256, len(angles_diff)])
    activity_measure_T = np.zeros([4, len(angles_diff)])

    angles_diff_T = angles_diff.T

    for i in range(0, 256):
        moving_vars[i] = np.var(rolling_window(angles_diff_T[i], 50), axis=-1)

    moving_vars[0:64] = np.sort(moving_vars[0:64], axis= 0)
    moving_vars[64:128] = np.sort(moving_vars[64:128], axis= 0)
    moving_vars[128:192] = np.sort(moving_vars[128:192], axis= 0)
    moving_vars[192:256] = np.sort(moving_vars[192:256], axis= 0)

    for i in range(0, 4):
        for j in range (0, baz):
            activity_measure_T[i] = np.add(moving_vars[i*64+j], activity_measure_T[i])

    return (activity_measure_T.T)/baz


def get_csi_from_packets_dwt(packets, wavelet="db1", ch_no=-1):
    packet_iqs = []
    stream_nos = []
    macs = []
    macs_groupped = []
    rssi_list = []
    frame_seq_nos = []
    timestamps = []
    timestamp_origin = -1
    _mac_list = mac_list
    if(ch_no == 36):
        _mac_list = mac_list_ch36
    elif(ch_no == 40):
        _mac_list = mac_list_ch40
    elif(ch_no == 44):
        _mac_list = mac_list_ch44
    elif(ch_no == 48):
        _mac_list = mac_list_ch48

    #for each packet in pcap:
    for packet in packets:
        #get data
        payload, mac, H_uint8, rx_time, frame_seq_no, stream_core_no, chanspec, chip_ver, rssi = parse_packet(packet)
        if mac in _mac_list:
            H_uint32 = np.frombuffer(H_uint8, dtype=np.uint32)
            Hout = unpack_float(10, 1, 0, 1, 9, 5, NFFT, H_uint32)
            packet_iqs.append(np.reshape(Hout, [len(H_uint32), 2]))
            stream_nos.append(stream_core_no)
            macs.append(mac)
            frame_seq_nos.append(frame_seq_no)
            rssi_list.append(rssi)
            timestamps.append(rx_time)
            if(timestamp_origin == -1):
                timestamp_origin = rx_time
            #print(f'mac:{mac}, frame: {frame_seq_no}, stream_core_no:{stream_core_no}, chansepc:{chanspec}')

    if(len(rssi_list) == 0):
        return None, None, None, None, None

    #re-format to np.complex
    packet_iqs_mat = np.concatenate([packet_iqs], axis=0)

    packet_iqs_reformatted = packet_iqs_mat.astype(np.float32).view(dtype=np.complex64)[:,:,0] * 1e-3
    # print(packet_iqs_reformatted.shape)
    a = pywt.wavedec(packet_iqs_reformatted, wavelet, level=8)
    packet_iqs_reformatted = []
    for i in a:
        packet_iqs_reformatted.append(i)
        if len(packet_iqs_reformatted) == 1:
            continue
        else:
            packet_iqs_reformatted = [np.concatenate(packet_iqs_reformatted, axis=1)]
    packet_iqs_reformatted =packet_iqs_reformatted[0]
    # packet_iqs_reformatted = np.fft.fftshift(packet_iqs_reformatted, axes=(1,))
    # print(packet_iqs_reformatted.shape)

    packet_iqs_ch1 = []
    packet_iqs_ch2 = []
    rssi_list_filtered = []
    timestamps_filtered = []

    for i in range(0, len(packet_iqs_reformatted)-1):
        if(macs[i] == macs[i+1] and frame_seq_nos[i] == frame_seq_nos[i+1] and stream_nos[i] != stream_nos[i+1]):
            macs_groupped.append(macs[i])
            timestamps_filtered.append(timestamps[i] - timestamp_origin)

            mask = spectrum_mask_dict[macs[i]]
            packet_iqs_reformatted[i] = np.multiply(packet_iqs_reformatted[i], mask)
            packet_iqs_reformatted[i+1] = np.multiply(packet_iqs_reformatted[i+1], mask)

            if(stream_nos[i] == 0):
                packet_iqs_ch1.append(packet_iqs_reformatted[i])
                packet_iqs_ch2.append(packet_iqs_reformatted[i+1])
                rssi_list_filtered.append(rssi_list[i])
            else:
                packet_iqs_ch1.append(packet_iqs_reformatted[i+1])
                packet_iqs_ch2.append(packet_iqs_reformatted[i])
                rssi_list_filtered.append(rssi_list[i+1])

    if(len(rssi_list_filtered) == 0):
        return None, None, None, None, None

    return packet_iqs_ch1, packet_iqs_ch2, rssi_list_filtered, macs_groupped, timestamps_filtered

def get_csi_from_packets_gesture_recognition (packets, filter_oob = False, agc_correction=True, threshold_sigma = 3):

    packet_iqs_ch1, packet_iqs_ch2, rssi_list_filtered, macs_groupped, timestamps = get_csi_from_packets_mimo (packets, filter_oob)
    if(packet_iqs_ch1 is None):
        return None, None, None


    #Get amplitudes and angles
    amplitudes_mat_ch1 = np.abs(packet_iqs_ch1)
    amplitudes_mat_ch2 = np.abs(packet_iqs_ch2)
    angles_mat_ch1 = np.angle(packet_iqs_ch1)
    angles_mat_ch2 = np.angle(packet_iqs_ch2)


    #amplitudes_mat_ch1 = dc_subcarrier_interpolate(amplitudes_mat_ch1)
    #amplitudes_mat_ch2 = dc_subcarrier_interpolate(amplitudes_mat_ch2)

    #mask noise
    for i in range (0, len(angles_mat_ch1)):
        if macs_groupped[i] in spectrum_mask_dict:
            mask = spectrum_mask_dict[macs_groupped[i]]
        else:
            print("Missing Mask!!!")

        angles_mat_ch1[i] = np.multiply(angles_mat_ch1[i], mask)
        angles_mat_ch2[i] = np.multiply(angles_mat_ch2[i], mask)

        if(filter_oob):
            amplitudes_mat_ch1[i] = np.multiply(amplitudes_mat_ch1[i], mask)
            amplitudes_mat_ch2[i] = np.multiply(amplitudes_mat_ch2[i], mask)

    angles_diff = np.subtract(angles_mat_ch1,angles_mat_ch2)
    angles_diff = CSI_angle_Interpolation(angles_diff)


    #phase wrap fix
    angles_diff_tmp = angles_diff.copy()
    np.add(math.pi *2, angles_diff, out = angles_diff, where= angles_diff_tmp < 0)

    angles_diff = np.unwrap(angles_diff, axis=0)
    #angles_diff = dc_subcarrier_interpolate(angles_diff)

    for i in range (1, len(angles_diff)):
        for j in range (0, len(angles_diff[0])):
            if((angles_diff[i-1][j] - angles_diff[i][j]) > 2.5):   #Not sure why I need this, but it fixes the problem
                angles_diff[i][j] = math.pi + angles_diff[i][j]
            elif((angles_diff[i-1][j] - angles_diff[i][j]) < -2.5):   #Not sure why I need this, but it fixes the problem
                angles_diff[i][j] = angles_diff[i][j] - math.pi


    #AGC fix
    amplitudes_mat_ch1 = CSI_Interpolation(3,amplitudes_mat_ch1)
    amplitudes_mat_ch2 = CSI_Interpolation(3,amplitudes_mat_ch2)
    if(agc_correction):
        amplitudes_mat_ch1, c = numerical_differentiation(amplitudes_mat_ch1)
        amplitudes_mat_ch1 = numerical_differentiation_outlier_removal_v1(amplitudes_mat_ch1,0.16)
        amplitudes_mat_ch1 = numerical_integration(amplitudes_mat_ch1, c)
        amplitudes_mat_ch2, c = numerical_differentiation(amplitudes_mat_ch2)
        amplitudes_mat_ch2 = numerical_differentiation_outlier_removal_v1(amplitudes_mat_ch2,0.16)
        amplitudes_mat_ch2 = numerical_integration(amplitudes_mat_ch2, c)

    angles_diff, c = numerical_differentiation(angles_diff)
    angles_diff = numerical_differentiation_outlier_removal_subcarrierwise(angles_diff, threshold_sigma = threshold_sigma)
    angles_diff = numerical_integration(angles_diff, c)

    return amplitudes_mat_ch1, amplitudes_mat_ch2, angles_diff, rssi_list_filtered, timestamps
