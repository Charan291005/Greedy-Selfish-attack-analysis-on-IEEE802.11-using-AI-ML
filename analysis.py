import os
import dpkt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from concurrent.futures import ProcessPoolExecutor

# Paths
input_directory = r"D:\kitsunet dataset\pcaps"
output_directory = "output_analysis"
os.makedirs(output_directory, exist_ok=True)

# Function to extract metrics from a single PCAP file
def process_pcap_file(filename, model=None, summary_data=[]):
    input_file = os.path.join(input_directory, filename)
    output_csv_file = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_results.csv")

    try:
        with open(input_file, 'rb') as f:
            reader = dpkt.pcap.Reader(f)
            packet_sizes, packet_freqs, device_ids = [], [], []

            for ts, buf in reader:
                try:
                    radio_tap = dpkt.radiotap.Radiotap(buf)
                    ieee80211 = radio_tap.data

                    if isinstance(ieee80211, dpkt.ieee80211.IEEE80211):
                        packet_sizes.append(len(buf))
                        packet_freqs.append(ts)
                        if ieee80211.type == 0:
                            src = ieee80211.mgmt.src
                        elif ieee80211.type == 1:
                            src = ieee80211.ra if hasattr(ieee80211, 'ra') else "Unknown"
                        elif ieee80211.type == 2:
                            src = ieee80211.data_frame
                        else:
                            src = "Unknown"
                        device_ids.append(src)
                except Exception:
                    continue

            if not packet_sizes:
                print(f"No valid 802.11 packets found in {filename}.")
                return

            # Calculate greedy metric
            greedy_metric = [(size / (freq + 1)) * 100 for size, freq in zip(packet_sizes, packet_freqs)]

            # Prepare DataFrame
            data = pd.DataFrame({
                'packet_size': packet_sizes,
                'packet_freq': packet_freqs,
                'greedy_metric': greedy_metric,
                'device_id': device_ids
            })

            # Apply the ML model for classification
            if model is not None:
                data['is_attack'] = model.predict(data[['packet_size', 'packet_freq', 'greedy_metric']])
            else:
                data['is_attack'] = 0  # No attack detection if model is not provided

            # Save results
            data.to_csv(output_csv_file, index=False)
            print(f"Results saved for {filename}")

            # Add to summary data
            total_packets = len(data)
            attacks_detected = data['is_attack'].sum()
            summary_data.append({
                'filename': filename,
                'total_packets': total_packets,
                'attacks_detected': attacks_detected
            })

            # Plot detected anomalies
            plot_anomalies(data, filename)
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Function to plot anomalies
def plot_anomalies(data, filename):
    attack_data = data[data['is_attack'] == 1]
    plt.figure(figsize=(10, 6))
    plt.scatter(data.index, data['greedy_metric'], label="Greedy Metric (Normal)", alpha=0.6)
    if not attack_data.empty:
        plt.scatter(attack_data.index, attack_data['greedy_metric'], color='red', label="Detected Attacks", alpha=0.8)
    plt.title(f"Attack Analysis for {filename}")
    plt.xlabel("Packet Index")
    plt.ylabel("Greedy Metric")
    plt.legend()
    plt.savefig(os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_anomalies.png"))
    plt.close()
    print(f"Anomaly plot saved for {filename}")

# Train ML model
def train_ml_model():
    # Simulated dataset for demonstration purposes
    # Replace this with actual pre-labeled training data
    df = pd.DataFrame({
        'packet_size': [100, 200, 500, 1000, 1500],
        'packet_freq': [0.1, 0.2, 0.5, 0.7, 1.0],
        'greedy_metric': [5000, 1000, 150, 100, 50],
        'is_attack': [0, 0, 1, 1, 1]
    })

    X = df[['packet_size', 'packet_freq', 'greedy_metric']]
    y = df['is_attack']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Training Complete:")
    print(classification_report(y_test, y_pred))

    return model

# Main execution
if __name__ == "__main__":
    print("Training ML model...")
    model = train_ml_model()

    print("Processing PCAP files...")
    pcap_files = [file for file in os.listdir(input_directory) if file.endswith(".pcap")]

    summary_data = []
    with ProcessPoolExecutor() as executor:
        for filename in pcap_files:
            executor.submit(process_pcap_file, filename, model=model, summary_data=summary_data)

    # Save summary results
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(output_directory, "summary_results.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary results saved to {summary_csv}")

    print("Processing complete.")
