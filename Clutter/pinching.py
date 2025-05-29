import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SleepApneaDetector:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.time = None
        self.body_pos = None
        self.pulse = None
        self.spo2 = None
        self.flow = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path, header=None)
        self.time = self.data[0].astype(float) / 1000  # Convert to seconds
        self.body_pos = self.data[1].astype(int)
        self.pulse = self.data[2].astype(float)
        self.spo2 = self.data[3].astype(float)
        self.flow = self.data[7].astype(float)

    def detect_apnea(self, flow_threshold=0.2, spo2_threshold=90, duration_threshold=10):
        # Convert time to differences
        time_diff = np.diff(self.time)
        mean_interval = np.mean(time_diff)
        
        # Find low flow periods
        apnea_mask = self.flow < flow_threshold
        low_flow_indices = np.where(apnea_mask)[0]
        
        # Identify contiguous low-flow segments
        apnea_events = []
        start_idx = low_flow_indices[0]
        
        for i in range(1, len(low_flow_indices)):
            if low_flow_indices[i] != low_flow_indices[i-1] + 1:
                end_idx = low_flow_indices[i-1]
                duration = (end_idx - start_idx + 1) * mean_interval
                if duration >= duration_threshold:
                    # Check for SpO2 drop
                    spo2_drops = self.spo2[start_idx:end_idx+1] < spo2_threshold
                    if np.any(spo2_drops):
                        apnea_events.append((self.time[start_idx], self.time[end_idx]))
                start_idx = low_flow_indices[i]
        
        # Add the last segment if it meets the criteria
        end_idx = low_flow_indices[-1]
        duration = (end_idx - start_idx + 1) * mean_interval
        if duration >= duration_threshold and np.any(self.spo2[start_idx:end_idx+1] < spo2_threshold):
            apnea_events.append((self.time[start_idx], self.time[end_idx]))
        
        return apnea_events

    def plot_data(self, apnea_events):
        plt.figure(figsize=(12, 8))
        plt.plot(self.time, self.flow, label='Flow')
        plt.plot(self.time, self.spo2, label='SpO2')
        plt.plot(self.time, self.pulse, label='Pulse')
        
        # Highlight apnea events
        for start, end in apnea_events:
            plt.axvspan(start, end, color='red', alpha=0.3, label='Apnea Event')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.legend()
        plt.title('Sleep Apnea Detection')
        plt.show()

# Example usage
detector = SleepApneaDetector('data.csv')
detector.load_data()
apnea_events = detector.detect_apnea()
detector.plot_data(apnea_events)
