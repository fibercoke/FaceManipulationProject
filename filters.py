import cv2
import numpy as np
import pykalman


class MeanFilter:
    def __init__(self, history_size):
        self.N = history_size
        self.history = []

    def correct(self, points):
        if len(self.history) < self.N:
            self.history.append(points)
        else:
            self.history.append(points)
            del self.history[0]

    def predict(self):
        ret = np.zeros_like(self.history[0])
        for x in self.history:
            ret += x
        return ret / self.N


class KalmanFilters:
    def __init__(self, filter_num):
        self.points = np.zeros(shape=(filter_num, 2))
        self.N = filter_num
        self.kalman_list = []
        self.filtered_state_means = np.nan * np.ones(shape=(filter_num, 4))
        self.filtered_state_covariances = (np.zeros(shape=(filter_num, 4, 4)))
        for i in range(filter_num):
            self.filtered_state_covariances[i] = np.eye(4)
            measurementMatrix = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0]], np.float32)

            transitionMatrix = np.array([[1, 0, 1, 0],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], np.float32)

            processNoiseCov = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32) * 0.03
            kalman = pykalman.KalmanFilter(observation_matrices=measurementMatrix,
                                           transition_matrices=transitionMatrix,
                                           transition_covariance=processNoiseCov)
            self.kalman_list.append(kalman)

    def correct(self, points):
        self.points = points
        for i in range(self.N):
            if np.all(np.isnan(self.filtered_state_means[i])):
                self.filtered_state_means[i] = np.hstack([points[i], np.zeros(2)])
            self.filtered_state_means[i], self.filtered_state_covariances[i] = self.kalman_list[i].filter_update(
                self.filtered_state_means[i], self.filtered_state_covariances[i], points[i])

    def predict(self):
        return np.vstack([self.filtered_state_means[:, :2], self.points[self.N:]])
