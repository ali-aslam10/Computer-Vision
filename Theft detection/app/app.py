from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import pickle
from PyQt5 import QtWidgets  # PyQt5 instead of PyQt4
from math import factorial
from tensorflow.keras.models import load_model
import numpy.matlib

# Seed for reproducibility
seed = 7
np.random.seed(seed)

# Load model from JSON


# Load weights from .mat file
def load_weights(model, weight_path):
    # dict2 = loadmat(weight_path)
    # weights_dict = conv_dict(dict2)
    # for i, layer in enumerate(model.layers):
    #     weights = weights_dict[str(i)]
    #     layer.set_weights(weights)
    # return model
    dict2 = loadmat(weight_path)
    dict_weights = conv_dict(dict2)

    for i, layer in enumerate(model.layers):
        if str(i) in dict_weights:
            weights = dict_weights[str(i)]
            layer.set_weights(weights)
    return model
       

# Convert loaded .mat file to usable weights dictionary
def conv_dict(dict2):
    dict_converted = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict_converted[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                dict_converted[str(i)] = [w[0] if w.shape in [(1, x) for x in range(5000)] else w for w in weights]
    return dict_converted

# Savitzky-Golay smoothing filter
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    window_size = np.abs(np.int32(window_size))
    order = np.abs(np.int32(order))
    
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("Window size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("Window size is too small for the polynomials order")
    
    half_window = (window_size - 1) // 2
    order_range = range(order + 1)
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

# Load Video Features
def load_dataset_One_Video_Features(Test_Video_Path):
    with open(Test_Video_Path, "r") as f:
        words = f.read().split()
    
    num_feat = len(words) // 4096
    VideoFeatues = np.float32(words[:4096])
    
    for feat in range(1, num_feat):
        feat_row = np.float32(words[feat * 4096:(feat + 1) * 4096])
        VideoFeatues = np.vstack((VideoFeatues, feat_row))
    
    return VideoFeatues

# PyQt5 Widget for GUI
class PrettyWidget(QtWidgets.QWidget):

    def __init__(self):
        super(PrettyWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(500, 100, 500, 500)
        self.setWindowTitle('Anomaly Detection')
        btn = QtWidgets.QPushButton('ANOMALY DETECTION SYSTEM \n Please select video', self)
        
        
        weights_path ='weights_L1L2.mat'
        model_path ='model.h5'
        
        # Load Model
        global model
        model = load_model(model_path)
        load_weights(model, weights_path)
        
        btn.resize(btn.sizeHint())
        btn.clicked.connect(self.SingleBrowse)
        btn.move(150, 200)
        self.show()

    def SingleBrowse(self):
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', "")
        cap = cv2.VideoCapture(video_path)
        
        Total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_segments = np.linspace(1, Total_frames, num=33).round()
        
        FeaturePath = video_path[:-4] + '.txt'
        inputs = load_dataset_One_Video_Features(FeaturePath)
        predictions = model.predict_on_batch(inputs)
        num_predictions = min(32, predictions.shape[0])
        Frames_Score = np.hstack([np.matlib.repmat(predictions[iv], 1, int(total_segments[iv + 1] - total_segments[iv])) for iv in range(num_predictions)])

        #Frames_Score = np.hstack([np.matlib.repmat(predictions[iv], 1, int(total_segments[iv + 1] - total_segments[iv])) for iv in range(32)])

        cap = cv2.VideoCapture(video_path)
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        print("Anomaly Prediction")
        x = np.linspace(1, Total_frames, int(Total_frames))
        scores1 = Frames_Score.reshape((Frames_Score.shape[1],))
        scores1 = savitzky_golay(scores1, 101, 3)
        
        plt.close()
        plt.axis([0, Total_frames, 0, 1])
        i = 0

        while True:
            flag, frame = cap.read()
            if flag:
                i += 1
                cv2.imshow('video', frame)
                
                if i % 25 == 1:
                    plot_length = min(i, len(scores1))
                    plt.plot(x[:plot_length], scores1[:plot_length], color='r', linewidth=3)
                    plt.xlabel('Frames')  # Label for x-axis
                    plt.ylabel('Anomaly Score')  # Label for y-axis
                    plt.title('Anomaly Detection Scores Over Time')  # Optional: Title for the plot
                    plt.draw()
                    plt.pause(1e-9)

                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                print(f"{pos_frame} frames")
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                cv2.waitKey(1000)

            if cv2.waitKey(10) == 27 or pos_frame == Total_frames:
                break

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = PrettyWidget()
    app.exec_()

if __name__ == "__main__":
    main()
