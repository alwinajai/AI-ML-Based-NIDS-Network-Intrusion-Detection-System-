# AI-ML-Based-NIDS-Network-Intrusion-Detection-System

**Real-time packet monitoring | Zero-day detection | Phishing detection | Automated reporting**

---

# Overview

Traditional signature-based IDS struggle to detect modern cyber threats because they rely on static attack definitions. This project solves that problem by building a hybrid Machine Learning-driven Network Intrusion Detection System (NIDS) capable of detecting both known and unknown attacks in real time.

The system integrates:

1. Supervised ML (Random Forest & XGBoost)
2. Unsupervised Anomaly Detection (Autoencoder)
3. Phishing URL Detection
4. Real-time Packet Capture (Scapy)
5. Threat Intelligence Mapping (CVSS & CVE)
6. Flet-based Desktop Dashboard
7. Automated PDF Session Reporting

---

# Key Features

## 1. Hybrid Intrusion Detection

✔ Detects known attacks using ML classifiers
✔ Detects zero-day attacks using Autoencoder anomaly detection
✔ Detects phishing URLs in live traffic

## 2. Real-Time Monitoring

✔ Live packet capture using Scapy
✔ Color-coded threat severity in GUI
✔ Clickable packet analysis panel
✔ Real-time statistics & attack counters

## 3. Threat Intelligence Mapping

Each malicious packet is enriched with:

* Attack category
* Severity level
* CVSS score
* Related CVEs
* Threat description

## 4. Automated PDF Reports

Generate professional session reports including:

* Attack statistics
* Protocol distribution charts
* Severity-colored packet logs
* Threat summaries and CVE references

---

# Real Time Pipeline

Live Packet Capture (Scapy)
↓
Feature Extraction + URL Detection
↓
Random Forest + XGBoost (Known Attacks)
↓
Autoencoder (Zero-Day Detection)
↓
Threat Mapping + Phishing Detection
↓
Flet GUI Dashboard
↓
PDF Report Generator

---

# Machine Learning Models

## Supervised Models

**Random Forest Classifier**
**XGBoost Classifier**

Detects:

* DoS / DDoS
* Brute-force attacks
* Port scanning
* Spoofing
* Botnet traffic

## Unsupervised Model — Autoencoder

The autoencoder is trained only on normal traffic.
If reconstruction error > threshold →
➡️ Packet flagged as Anomaly (Possible Zero-Day Attack)

---

# 🎣 Phishing URL Detection

For packets containing URLs, the system checks:

* Suspicious domain patterns
* URL entropy and length
* Special characters & keywords
* Blacklist heuristics

Phishing packets are highlighted RED in the UI for quick prioritization.

---

# GUI Features (Flet)

* Real-time scrolling packet table
* Click packet → detailed analysis panel
* Severity color coding

  * 🟢 Benign
  * 🟠 Suspicious / Anomaly
  * 🔴 Critical / Phishing
* Attack counter & live statistics
* Export packet JSON
* Generate PDF report with custom save location

---

# Report Generation

PDF reports include:

## 1️⃣ Cover Page

* Packet count
* Attack count
* Anomaly count
* Phishing alerts
* Charts & statistics

## 2️⃣ Detailed Packet Logs

## 3️⃣ Severity-colored tables

## 4️⃣ Threat Intelligence Summary

---

# 🛠️ Tech Stack

## Programming

Python

## Networking

Scapy

## Machine Learning

Scikit-learn
XGBoost
TensorFlow / PyTorch (Autoencoder)

## UI

Flet

## Reporting & Visualization

ReportLab
Matplotlib

---

# Project Structure

```
NIDS_Project/
│
├── data/
├── models/
├── reports/
├── src/
│   ├── gui.py
│   ├── live_capture.py
│   ├── threat_mapping.py
│   ├── report_generator.py
│   ├── model_training.py
│   └── data_preprocessing.py
│
├── main.py
└── requirements.txt
```

---

# Setup & Virtual Environment

Follow these steps to run the project locally.

## 1️⃣ Clone the Repository

```
git clone https://github.com/alwinajai/AI-ML-Based-NIDS-Network-Intrusion-Detection-System-.git
cd NIDS_Project
```

## 2️⃣ Create Virtual Environment

**Windows**

```
python -m venv nids_env
nids_env\Scripts\activate
```

**Linux / macOS**

```
python3 -m venv nids_env
source nids_env/bin/activate
```

## 3️⃣ Install Dependencies

```
pip install scapy flet scikit-learn xgboost matplotlib reportlab pandas numpy
```

## 4️⃣ Run the Application

1. Data Should Be Pre processed
```
#in the virtual enviornment enabled
python src\new_preprocess.py

```
2. Model Should Be Trained
```
python src\train_autoencoder.py
```
   
3. Configure Autoencoder
```
python src\optimized_autoencoder_config.py
```
  
6. Run the GUI code
```
python src\gui2.py
```

# Applications

* Cybersecurity research
* SOC training environments
* Small enterprise network monitoring
* Digital forensics & incident response
* Critical infrastructure monitoring

---

# Conclusion

This project delivers a production-ready ML-powered Network Intrusion Detection System combining real-time packet monitoring, hybrid intrusion detection, phishing analysis, and automated reporting in a single desktop application.
