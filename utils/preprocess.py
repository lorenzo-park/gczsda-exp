import warnings
warnings.filterwarnings("ignore")

from unpack.unpackfloat import unpack_float
from scapy.all import *
#from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy import signal
import cv2
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
    spectrogram_out = np.zeros([len(spectrogram)-1, len(spectrogram[0])], dtype=spectrogram.dtype) #1000, 256

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

def numerical_differentiation_outlier_removal_v2(spectrogram_d, sigma):
    mean = np.mean(spectrogram_d,axis=0)
    std = np.std(spectrogram_d,axis=0)
    outlier_thres = mean + (std*sigma)
    amplitudes_mat = spectrogram_d

    for i in range (0, len(amplitudes_mat)):
        comparison = np.abs(amplitudes_mat[i]) < outlier_thres
        amplitudes_mat[i] = amplitudes_mat[i] *(comparison)
        amplitudes_mat[i] = amplitudes_mat[i] + (amplitudes_mat[i-1] *(~comparison))

    return amplitudes_mat

def numerical_differentiation_outlier_removal_subcarrierwise(spectrogram_d, threshold_sigma = 3):
    spectrogram_out = spectrogram_d    #1000 x 256
    foo = int(len(spectrogram_out[0])/64)
    means = np.mean(spectrogram_d, axis=0)
    stds = np.std(spectrogram_d, axis=0)
    thres = np.add(means, stds*threshold_sigma)

    for i in range(0, len(spectrogram_d[0])):
        spectrogram_out[:, i] = np.where(spectrogram_d[:, i] > thres[i], 0, spectrogram_d[:, i])

    return spectrogram_out

def outlier_removal_subcarrierwise(spectrogram, threshold_sigma = 3):
    spectrogram_out = np.zeros(np.shape(spectrogram), dtype=spectrogram.dtype)    #1000 x 256
    means = np.mean(spectrogram, axis=0)
    stds = np.std(spectrogram, axis=0)
    thres_upper = np.add(means, stds*threshold_sigma)
    thres_lower = np.add(means, stds*threshold_sigma*-1)
    foo = int(len(spectrogram_out[0])/64)

    spectrogram_out[0] = spectrogram[0]
    outlier_cnt = 0
    for i in range(1, len(spectrogram)):
        for j in range (0, foo):
            subcarrier_start = j * 64
            subcarrier_end = j * 64 + 64 #python quirk

            is_outlier_upper = np.sum((spectrogram[i][subcarrier_start:subcarrier_end] - thres_upper[subcarrier_start:subcarrier_end]) > 0) > 16
            is_outlier_lower = np.sum((thres_lower[subcarrier_start:subcarrier_end] - spectrogram[i][subcarrier_start:subcarrier_end]) > 0) > 16
            is_outlier = is_outlier_upper or is_outlier_lower
            if(is_outlier):
                outlier_cnt += 1

            spectrogram_out[i][subcarrier_start:subcarrier_end] = spectrogram_out[i-1][subcarrier_start:subcarrier_end] if is_outlier else spectrogram[i][subcarrier_start:subcarrier_end]

    #print(outlier_cnt)
    return spectrogram_out

#reorders channels
def reorder_channels(spectrogram, order):
    spectrogram_out = np.zeros([len(spectrogram), 64*len(order)]) #1000, 256

    for i in range(0, len(spectrogram)):
        for channel in range(0, len(order)):
            spectrogram_out[i][channel*64:channel*64+64] = spectrogram[i][order[channel]*64:order[channel]*64+64]

    return spectrogram_out

#masks channels
def mask_channels(spectrogram, mask):
    spectrogram_out = np.zeros([len(spectrogram), 64*len(mask)]) #1000, 256

    for i in range(0, len(spectrogram)):
        for channel in range(0, 4):
            if(mask[channel]):
                spectrogram_out[i][channel*64:channel*64+64] = spectrogram[i][channel*64:channel*64+64]

    return spectrogram_out

#timeshift
#TODO: verify (may have off-by-one bugs; didn't carefully think)
def timeshift(spectrogram, shift_samples): #(not that timeshift)
    spectrogram_out = np.zeros([len(spectrogram), len(spectrogram[0])]) #1000, 256

    if(shift_samples < 0):
        for i in range(0, len(spectrogram) + shift_samples):
            spectrogram_out[i] = spectrogram[i - shift_samples]
        for i in range (len(spectrogram) + shift_samples, len(spectrogram)):
            spectrogram_out[i] = spectrogram_out[i-1]
    elif(shift_samples > 0):
        for i in range (0, shift_samples):
            spectrogram_out[i] = spectrogram[0]
        cnt = 0
        for i in range(shift_samples, len(spectrogram)):
            spectrogram_out[i] = spectrogram[cnt]
            cnt += 1
    else:
        return spectrogram
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
    if(len(spectrogram) < 2):
        return spectrogram

    #CSI Interpolation
    psd = np.mean(np.abs(spectrogram), axis=0)
    mean_level_per_channel  = np.zeros(4)
    threshold = np.zeros(4)

    for i in range (0, 4):
        for j in range (0, 64):
            mean_level_per_channel [i] += psd[i*64 + j]
        threshold[i] = mean_level_per_channel [i] / threshold_factor;

    #forward
    for i in range (1, len(spectrogram)):
        tmp = np.abs(spectrogram[i])
        for j in range (0, 4):
            if(sum(tmp[j*64:j*64+64]) < threshold[j]):
                spectrogram[i][j*64:j*64+64] = spectrogram[i-1][j*64:j*64+64]

    #backward
    for i in range (int(len(spectrogram)/4), -1, -1):
        tmp = np.abs(spectrogram[i])
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
    norm_max=None, norm_min=None, eps=1e-6, dtype=np.uint8
):
    if(len(X) == 0 or len(X[0]) == 0):
        return np.stack([X, X, X], axis=-1)

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
        V = V.astype(dtype)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=dtype)
    return V

#TODO fix (very, very poor implementation...)
def not_so_mono_to_color(
    X1: np.ndarray,X2: np.ndarray,X3: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6, interpolation=cv2.INTER_NEAREST, img_size_x=256, img_size_y=256
):

    V1 = mono_to_color(X1, mean, std, norm_max, norm_min, eps)
    V1 = cv2.resize(V1, (img_size_x, img_size_y),interpolation=interpolation)

    V2 = mono_to_color(X2, mean, std, norm_max, norm_min, eps)
    V2 = cv2.resize(V2, (img_size_x, img_size_y),interpolation=interpolation)

    V3 = mono_to_color(X3, mean, std, norm_max, norm_min, eps)
    V3 = cv2.resize(V3, (img_size_x, img_size_y),interpolation=interpolation)

    V1 = V1.T
    V1[1] = V2.T[0]
    V1[2] = V3.T[0]
    V1 = V1.T

    return V1


def resize_timeaware_linear(amplitudes_mat, timestamps, IMG_SIZE, TOTAL_TIME, START_TIME = 0.0):
    out = np.zeros([IMG_SIZE, len(amplitudes_mat[0])], dtype=amplitudes_mat[0][0].dtype)
    timestamps = np.array(timestamps)
    for i in range(0, IMG_SIZE):
        t = TOTAL_TIME * (float(i) / IMG_SIZE) + START_TIME
        idx = (np.abs(timestamps - t)).argmin()

        if(timestamps[idx] - t) > 0:
            idx_prev = max(idx-1, 0)
            idx_next = idx
        else:
            idx_prev = idx
            idx_next = min(idx+1, len(timestamps)-1)

        time_prev = timestamps[idx_prev]
        time_next = timestamps[idx_next]

        if(time_prev == time_next):
            out[i] = amplitudes_mat[idx]

        else:
            time_len = time_next - time_prev
            offset_ratio = (t - time_prev) /time_len
            out[i] = (amplitudes_mat[idx_prev] * (1-offset_ratio)) + (amplitudes_mat[idx_next] * offset_ratio)
            '''
            if(np.sum(out[i] < 0) > 0):
                print("this shouldn't happen:")
                print(f"idx_prev:{idx_prev}, idx_next:{idx_next}, time_prev:{time_prev}, time_next:{time_next}, t:{t}, offset_ratio:{offset_ratio}")
            '''


    return out

def resize_timeaware_nearest(amplitudes_mat, timestamps, IMG_SIZE_X, IMG_SIZE_Y, TOTAL_TIME, START_TIME = 0.0):
    out = np.zeros([IMG_SIZE_X, IMG_SIZE_Y], dtype=amplitudes_mat[0][0].dtype)
    timestamps = np.array(timestamps)
    for i in range(0, IMG_SIZE_X):
        t = TOTAL_TIME * (float(i) / IMG_SIZE_X) + START_TIME
        idx = (np.abs(timestamps - t)).argmin()
        out[i] = amplitudes_mat[idx]

    return out


def resize_timeaware(amplitudes_mat, timestamps, IMG_SIZE, TOTAL_TIME, START_TIME = 0.0, mode = 'linear'):
    if(mode.lower() == 'linear'):
        return resize_timeaware_linear(amplitudes_mat, timestamps, IMG_SIZE, TOTAL_TIME, START_TIME)
    elif(mode.lower() == 'nearest'):
        return resize_timeaware_nearest(amplitudes_mat, timestamps, IMG_SIZE, IMG_SIZE, TOTAL_TIME, START_TIME)
    else:
        print("Where did you learn to fly")
        return None

def resize_nearest(amplitudes_mat, IMG_SIZE):
    out = np.zeros([IMG_SIZE, IMG_SIZE])
    for i in range(0, IMG_SIZE):
        idx = int(len(amplitudes_mat) * (float(i) / IMG_SIZE))
        out[i] = amplitudes_mat[idx]

    return out

def parse_packet(packet):
    rx_time = packet.time
    payload = bytes(packet[UDP].payload)
    rssi = np.frombuffer(payload, dtype=np.int8)[3]
    mac = bytes(np.frombuffer(payload, dtype=np.uint8)[4:10])

    length = len(bytes(np.frombuffer(payload, dtype=np.uint8)))
    NFFT = -1
    if(length == 1042):
        NFFT = 256
    elif (length == 530):
        NFFT = 128
    elif (length == 274):
        NFFT = 64
    else:
        return None, None, None, None, None, None, None, None, None

    H_uint8 = bytes(np.frombuffer(payload, dtype=np.uint8)[HOFFSET+2:HOFFSET+2+NFFT*4])
    frame_seq_no = np.frombuffer(payload, dtype=np.uint16)[5]
    stream_core_no = np.frombuffer(payload, dtype=np.uint16)[6]
    chanspec = np.frombuffer(payload, dtype=np.uint16)[7]
    chip_ver = np.frombuffer(payload, dtype=np.uint16)[8]

    return payload, mac, H_uint8, rx_time, frame_seq_no, stream_core_no, chanspec, chip_ver, rssi, NFFT


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

'''
Alternative:
def IQ_to_polar_coordinate(packet_iqs, is_degree=True):
    packet_iqs_mat = np.concatenate([packet_iqs], axis=0)

    packet_iqs_reformatted = packet_iqs_mat.astype(np.float32).view(dtype=np.complex64)[:,:,0] * 1e-3
    packet_iqs_reformatted = np.fft.fftshift(packet_iqs_reformatted, axes=(1,))
    amplitudes_mat = np.abs(packet_iqs_reformatted )
    angles_mat = np.angle(packet_iqs_reformatted, deg=is_degree)

    return amplitudes_mat, angles_mat
'''

def get_csi_from_packets(packets, antenna_no = 0, filter_oob = False, do_pad = True, pad_dir = 0):
    packet_iqs = []
    macs = []
    rssi_list = []
    timestamps = []
    timestamp_origin = -1
    for packet in packets:
        payload, mac, H_uint8, rx_time, frame_seq_no, stream_core_no, chanspec, chip_ver, rssi, NFFT = parse_packet(packet)
        if(antenna_no == stream_core_no):
            H_uint32 = np.frombuffer(H_uint8, dtype=np.uint32)
            Hout = unpack_float(10, 1, 0, 1, 9, 5, NFFT, H_uint32)

            if(len(Hout) < 512 and do_pad):
                padding = np.zeros([512-len(Hout)])
                if(pad_dir == 0):
                    Hout = np.concatenate((Hout,padding), axis=None)
                    NFFT = 256
                else:
                    Hout = np.concatenate((padding, Hout), axis=None)
                    NFFT = 256
            if(timestamp_origin == -1):
                timestamp_origin = rx_time

            timestamps.append(rx_time - timestamp_origin)
            packet_iqs.append(np.reshape(Hout, [NFFT, 2]))
            macs.append(mac)
            rssi_list.append(rssi)

    channel_list = []

    if(len(packet_iqs) == 0):
        return np.zeros([0, 256]), np.zeros([0, 256]), [], [], []

    else:
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

    return amplitudes_mat, angles_mat, channel_list, rssi_list, timestamps


def get_csi_from_packets_mimo (packets, filter_oob = False, ch_no = -1, do_pad = True, pad_dir = 0, verbose=False):
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
        payload, mac, H_uint8, rx_time, frame_seq_no, stream_core_no, chanspec, chip_ver, rssi, NFFT = parse_packet(packet)
        if mac in _mac_list:
            H_uint32 = np.frombuffer(H_uint8, dtype=np.uint32)
            Hout = unpack_float(10, 1, 0, 1, 9, 5, NFFT, H_uint32)

            if(len(Hout) < 512 and do_pad):
                padding = np.zeros([512-len(Hout)])
                if(pad_dir == 0):
                    Hout = np.concatenate((Hout,padding), axis=None)
                    NFFT = 256

                else:
                    Hout = np.concatenate((padding, Hout), axis=None)
                    NFFT = 256


            packet_iqs.append(np.reshape(Hout, [NFFT, 2]))
            stream_nos.append(stream_core_no)
            macs.append(mac)
            frame_seq_nos.append(frame_seq_no)
            rssi_list.append(rssi)
            timestamps.append(rx_time)
            if(timestamp_origin == -1):
                timestamp_origin = rx_time
            #print(f'mac:{mac}, frame: {frame_seq_no}, stream_core_no:{stream_core_no}, chansepc:{chanspec}')

    if(len(rssi_list) == 0):
        if(verbose):
            print("Warning: no CSI data detected (case 1)")
        return None, None, None, None, None

    #re-format to np.complex
    packet_iqs_mat = np.concatenate([packet_iqs], axis=0)

    packet_iqs_reformatted = packet_iqs_mat.astype(np.float32).view(dtype=np.complex64)[:,:,0] * 1e-3
    packet_iqs_reformatted = np.fft.fftshift(packet_iqs_reformatted, axes=(1,))

    #oob filter
    if(filter_oob):
        for i in range (0, len(packet_iqs_reformatted)):
            if macs[i] in spectrum_mask_dict:
                mask = spectrum_mask_dict[macs[i]]
            else:
                print("Missing Mask!!!")

            packet_iqs_reformatted[i] = np.multiply(packet_iqs_reformatted[i], mask)

    #group
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
        print("Warning: no CSI data detected (case 2)")
        return None, None, None, None, None

    return packet_iqs_ch1, packet_iqs_ch2, rssi_list_filtered, macs_groupped, timestamps_filtered


def get_csi_from_packets_aoa (packets, ch_no, filter_oob = False , verbose=False):

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
        if(verbose):
            print("rssi_list_ch1 is None")
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

def get_ch(spectrogram, ch_pos, remove_guardbands = False):
    out = spectrogram.T[ch_pos*64:ch_pos*64+64].T
    if(remove_guardbands):
        out = out.T[0+6:64-5].T

    return out

def get_csi_from_packets_gesture_recognition (packets, filter_oob = False, agc_correction=True, threshold_sigma = 3, dc_subcarrier_interpolation = False):

    packet_iqs_ch1, packet_iqs_ch2, rssi_list_filtered, macs_groupped, timestamps = get_csi_from_packets_mimo (packets, filter_oob)
    if(packet_iqs_ch1 is None):
        print("packet_iqs_ch1 is None")
        return None, None, None, None, None


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


    #amplitude interpolation
    amplitudes_mat_ch1 = CSI_Interpolation(3,amplitudes_mat_ch1)
    amplitudes_mat_ch2 = CSI_Interpolation(3,amplitudes_mat_ch2)

    #Amplitude AGC fix
    if(agc_correction):
        amplitudes_mat_ch1, c = numerical_differentiation(amplitudes_mat_ch1)
        amplitudes_mat_ch1 = numerical_differentiation_outlier_removal_v1(amplitudes_mat_ch1,0.16)
        amplitudes_mat_ch1 = numerical_integration(amplitudes_mat_ch1, c)
        amplitudes_mat_ch2, c = numerical_differentiation(amplitudes_mat_ch2)
        amplitudes_mat_ch2 = numerical_differentiation_outlier_removal_v1(amplitudes_mat_ch2,0.16)
        amplitudes_mat_ch2 = numerical_integration(amplitudes_mat_ch2, c)

    #angles outlier removal
    if(len(angles_diff) > 5):
        angles_diff, c = numerical_differentiation(angles_diff)
        angles_diff = numerical_differentiation_outlier_removal_subcarrierwise(angles_diff, threshold_sigma = threshold_sigma)
        angles_diff = numerical_integration(angles_diff, c)

    #dc subcarrier interp
    if(dc_subcarrier_interpolation):
        angles_diff = dc_subcarrier_interpolate(angles_diff)
        amplitudes_mat_ch1 = dc_subcarrier_interpolate(amplitudes_mat_ch1)
        amplitudes_mat_ch2 = dc_subcarrier_interpolate(amplitudes_mat_ch2)

    return amplitudes_mat_ch1, amplitudes_mat_ch2, angles_diff, rssi_list_filtered, timestamps

#original:
#DBINV Convert from decibels (c) 2008-2011 Daniel Halperin <dhalperi@cs.washington.edu>
#see https://github.com/dhalperi/linux-80211n-csitool-supplementary/tree/master/matlab
def dbinv(x):
    return 10**(x/10)

#original:
#GET_SCALED_CSI Converts a CSI struct to a channel matrix H.
#see https://github.com/dhalperi/linux-80211n-csitool-supplementary/tree/master/matlab
#signficiantly modified to meet my use cases
#!!!!!!!!!!only to be used by get_csi_from_packets_mimo_single_channel!!!!!!!!

def get_scaled_csi(csi, rssi, ignore_rssi=False):

    csi_sq = np.abs(csi)**2
    csi_pwr = np.mean(csi_sq)
    rssi_pwr = dbinv(rssi)

    if(ignore_rssi):
        scale = 1 / csi_pwr;
    else:
        scale = rssi_pwr / csi_pwr;

    if(np.sum(np.isnan(csi * np.sqrt(scale))) > 0):
        print(csi_pwr)
        print(rssi_pwr)

    return csi * np.sqrt(scale)


#for data with only one active channel
def get_csi_from_packets_mimo_single_channel (packets, filter_oob = False, threshold_sigma = 3, dc_subcarrier_interpolation = False, use_decibel = False, remove_guardbands = True, sort_packets_timestamp =True):

    packet_iqs_ch1, packet_iqs_ch2, rssi_list_filtered, macs_groupped, timestamps = get_csi_from_packets_mimo (packets, filter_oob)
    if(packet_iqs_ch1 is None):
        print("packet_iqs_ch1 is None")
        return None, None, None, None, None

    #apparantly some data files have packets with
    if(sort_packets_timestamp):
        inds = np.argsort(timestamps)
        #packet_iqs_ch1 = packet_iqs_ch1[packet_order]
        #packet_iqs_ch2 = packet_iqs_ch2[packet_order]
        packet_iqs_ch1 = [packet_iqs_ch1[idx] for idx in inds]
        packet_iqs_ch2 = [packet_iqs_ch2[idx] for idx in inds]
        rssi_list_filtered = [rssi_list_filtered[idx] for idx in inds]
        macs_groupped = [macs_groupped[idx] for idx in inds]
        timestamps = [timestamps[idx] for idx in inds]



    #Get amplitudes and angles
    amplitudes_mat_ch1 = np.abs(packet_iqs_ch1)
    amplitudes_mat_ch2 = np.abs(packet_iqs_ch2)
    angles_mat_ch1 = np.angle(packet_iqs_ch1)
    angles_mat_ch2 = np.angle(packet_iqs_ch2)

    ch_num = mac_dict[macs_groupped[0]]
    ch_pos = int((ch_num-36)/4)

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


    #angles outlier removal
    if(len(angles_diff) > 10):
        angles_diff, c = numerical_differentiation(angles_diff)
        angles_diff = numerical_differentiation_outlier_removal_subcarrierwise(angles_diff, threshold_sigma = threshold_sigma)
        angles_diff = numerical_integration(angles_diff, c)

    #dc subcarrier interp
    if(dc_subcarrier_interpolation):
        angles_diff = dc_subcarrier_interpolate(angles_diff)
        amplitudes_mat_ch1 = dc_subcarrier_interpolate(amplitudes_mat_ch1)
        amplitudes_mat_ch2 = dc_subcarrier_interpolate(amplitudes_mat_ch2)

    #dB
    if(use_decibel):
        amplitudes_mat_ch1 = 10*np.log10(amplitudes_mat_ch1 + 1e-20)
        amplitudes_mat_ch2 = 10*np.log10(amplitudes_mat_ch2 + 1e-20)

    angles_diff = get_ch(angles_diff, ch_pos, remove_guardbands = remove_guardbands)
    amplitudes_mat_ch1 = get_ch(amplitudes_mat_ch1, ch_pos, remove_guardbands = remove_guardbands)
    amplitudes_mat_ch2 = get_ch(amplitudes_mat_ch2, ch_pos, remove_guardbands = remove_guardbands)

    return amplitudes_mat_ch1, amplitudes_mat_ch2, angles_diff, rssi_list_filtered, timestamps


def get_csi_angle_diff (packets, filter_oob = False, threshold_sigma = 3):

    packet_iqs_ch1, packet_iqs_ch2, rssi_list_filtered, macs_groupped, timestamps = get_csi_from_packets_mimo (packets, filter_oob)
    if(packet_iqs_ch1 is None):
        print("packet_iqs_ch1 is None")
        return None, None, None

    #Get amplitudes and angles
    angles_mat_ch1 = np.angle(packet_iqs_ch1)
    angles_mat_ch2 = np.angle(packet_iqs_ch2)

    #mask noise
    for i in range (0, len(angles_mat_ch1)):
        if macs_groupped[i] in spectrum_mask_dict:
            mask = spectrum_mask_dict[macs_groupped[i]]
        else:
            print("Missing Mask!!!")

        angles_mat_ch1[i] = np.multiply(angles_mat_ch1[i], mask)
        angles_mat_ch2[i] = np.multiply(angles_mat_ch2[i], mask)

    angles_diff = np.subtract(angles_mat_ch1,angles_mat_ch2)
    angles_diff = CSI_angle_Interpolation(angles_diff)

    #resolve ambiguities & phase wrapping
    angles_diff_tmp = angles_diff.copy()
    np.add(math.pi *2, angles_diff, out = angles_diff, where= angles_diff_tmp < 0)
    angles_diff = dc_subcarrier_interpolate(angles_diff)

    for i in range (1, len(angles_diff)):
        for j in range (0, len(angles_diff[0])):
            if((angles_diff[i-1][j] - angles_diff[i][j]) > 2.5):   #Not sure why I need this, but it fixes the problem
                angles_diff[i][j] = math.pi + angles_diff[i][j]
            elif((angles_diff[i-1][j] - angles_diff[i][j]) < -2.5):   #Not sure why I need this, but it fixes the problem
                angles_diff[i][j] = angles_diff[i][j] - math.pi

    '''
    #phase wrap fix
    angles_diff = np.unwrap(angles_diff, axis=0)

    '''

    #outlier removal
    angles_diff = outlier_removal_subcarrierwise(angles_diff, threshold_sigma = threshold_sigma)

    return angles_diff, rssi_list_filtered, timestamps


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