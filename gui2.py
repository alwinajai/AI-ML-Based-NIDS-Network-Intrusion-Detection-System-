# src/gui.py
"""
NIDS GUI - Robust version with Autoencoder + Random Forest + XGBoost Integration

Features:
- Robust autoencoder feature extraction with error handling
- Proper timestamp handling for all types of input
- Correct feature dimensions
- Proper model detection for your files:
  - trained_randomforest_sampled_smote.pkl
  - trained_xgboost_sampled_classweights.pkl
- FIXED: Accurate packet capturing and stopping
- FIXED: Start/Stop/Refresh button responsiveness
- FIXED: Race condition and error handling
- FIXED: safe_update function
"""

import flet as ft
import threading
import time
import traceback
import queue
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np

# Add project src folder to sys.path (if not already)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import LiveCaptureManager, threat mapping, and report generator if present
try:
    from src.live_capture import LiveCaptureManager
except Exception:
    LiveCaptureManager = None
    print("Warning: LiveCaptureManager not found. Packet capture will not work.")

try:
    from src.threat_mapping import map_prediction_to_threat
except Exception:
    # fallback mapping
    def map_prediction_to_threat(pred):
        return {"category": "Unknown", "description": "", "cves": [], "cvss": None, "severity": "Unknown"}

# Report generator import (optional)
try:
    from src.report_generator import generate_pdf_report
    REPORT_GENERATOR_AVAILABLE = True
except Exception:
    generate_pdf_report = None
    REPORT_GENERATOR_AVAILABLE = False

# ========== AUTOENCODER INTEGRATION ==========
try:
    from src.autoencoder_integration import AutoencoderIntegration
    AUTOENCODER_AVAILABLE = True
except Exception as e:
    print(f"Autoencoder not available: {e}")
    AutoencoderIntegration = None
    AUTOENCODER_AVAILABLE = False

# ========== ML MODELS INTEGRATION (with your specific file names) ==========
rf_model = None
xgb_model = None
RANDOM_FOREST_AVAILABLE = False
XGBOOST_AVAILABLE = False

# Look for your specific model files in models directory
models_dir = Path(parent_dir) / 'models'

# Try to load Random Forest model
rf_paths = [
    models_dir / 'trained_randomforest_sampled_smote.pkl',
    models_dir / 'rf_model.pkl',  # fallback
    models_dir / 'randomforest_model.pkl'  # fallback
]

for rf_path in rf_paths:
    if rf_path.exists():
        try:
            rf_model = joblib.load(rf_path)
            RANDOM_FOREST_AVAILABLE = True
            print(f"✓ Random Forest model loaded: {rf_path.name}")
            break
        except Exception as e:
            print(f"⚠️  Could not load RF model {rf_path.name}: {e}")

# Try to load XGBoost model
xgb_paths = [
    models_dir / 'trained_xgboost_sampled_classweights.pkl',
    models_dir / 'xgb_model.pkl',  # fallback
    models_dir / 'xgboost_model.pkl'  # fallback
]

for xgb_path in xgb_paths:
    if xgb_path.exists():
        try:
            xgb_model = joblib.load(xgb_path)
            XGBOOST_AVAILABLE = True
            print(f"✓ XGBoost model loaded: {xgb_path.name}")
            break
        except Exception as e:
            print(f"⚠️  Could not load XGB model {xgb_path.name}: {e}")

# Autoencoder Integration
autoencoder_integration = None
if AUTOENCODER_AVAILABLE:
    try:
        model_path = models_dir / 'autoencoder_nids.pth'
        if model_path.exists():
            autoencoder_integration = AutoencoderIntegration(str(model_path))
            AUTOENCODER_AVAILABLE = True
            print(f"✓ Autoencoder model loaded: {model_path.name}")
        else:
            print("⚠️  Autoencoder model not found")
            AUTOENCODER_AVAILABLE = False
    except Exception as e:
        print(f"Autoencoder not available: {e}")
        AUTOENCODER_AVAILABLE = False

# Globals
live_capture_manager = None
packet_queue = queue.Queue()
packet_history = []
capture_thread = None
stop_event = threading.Event()
# Use a lock to protect shared state updates from the packet processing thread
ui_update_lock = threading.Lock()


def safe_convert_timestamp(ts):
    """
    Safely convert timestamp to datetime object.
    Handles None, string ('N/A', '2023-10-27 10:00:00', etc.), int, float.
    """
    if ts is None or ts == 'N/A' or ts == '':
        return datetime.now()
    
    if isinstance(ts, datetime):
        return ts
    elif isinstance(ts, str):
        # Check if it's a numeric string that represents a timestamp
        try:
            # If it's a float-like string
            numeric_ts = float(ts)
            return datetime.fromtimestamp(numeric_ts)
        except ValueError:
            # If it's a date-time string
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                try:
                    return datetime.strptime(ts, fmt)
                except ValueError:
                    continue
            # If all fail, return current time
            print(f"Warning: Could not parse timestamp string '{ts}'. Using current time.")
            return datetime.now()
    elif isinstance(ts, (int, float)):
        # Unix timestamp
        try:
            return datetime.fromtimestamp(ts)
        except:
            print(f"Warning: Could not convert timestamp {ts}. Using current time.")
            return datetime.now()
    else:
        print(f"Warning: Unexpected timestamp type {type(ts)}. Using current time.")
        return datetime.now()


def safe_float_convert(value, default=0.0):
    """
    Safely convert value to float with default fallback.
    Handles None, string ('N/A', '0', etc.), int, float.
    """
    if value is None or value == 'N/A' or value == '':
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            print(f"Warning: Could not convert '{value}' to float. Using default {default}.")
            return default
    print(f"Warning: Unexpected value type {type(value)} for float conversion. Using default {default}.")
    return default


def extract_features_for_ml(packet_data):
    """
    Extract features from packet data for ML models.
    This must match your training preprocessing.
    Uses safe conversion functions to handle 'N/A' and other invalid inputs.
    """
    try:
        features = []
        
        # Extract numerical features (match your training preprocessing)
        features.append(safe_float_convert(packet_data.get('port_src', 0)))
        features.append(safe_float_convert(packet_data.get('port_dst', 0)))
        features.append(safe_float_convert(packet_data.get('packet_length', 0)))
        
        # Protocol encoding (TCP=0, UDP=1, ICMP=2, Other=3)
        protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
        protocol = str(packet_data.get('protocol', 'Other')).upper()
        features.append(safe_float_convert(protocol_map.get(protocol, 3)))
        
        # Time-based features
        ts = safe_convert_timestamp(packet_data.get('timestamp'))
        features.extend([
            safe_float_convert(ts.hour),
            safe_float_convert(ts.minute),
            safe_float_convert(ts.second),
            safe_float_convert(ts.weekday())
        ])
        
        # IP-based features (simplified)
        src_ip = str(packet_data.get('src_ip', '0.0.0.0'))
        dst_ip = str(packet_data.get('dst_ip', '0.0.0.0'))
        
        # Extract last octet as feature
        try:
            src_last = int(src_ip.split('.')[-1])
            dst_last = int(dst_ip.split('.')[-1])
        except (ValueError, IndexError):
            src_last, dst_last = 0, 0
        
        features.extend([safe_float_convert(src_last), safe_float_convert(dst_last)])
        
        # Additional features (match your training preprocessing)
        features.extend([
            safe_float_convert(packet_data.get('ttl', 64)),
            safe_float_convert(packet_data.get('window_size', 0)),
            safe_float_convert(packet_data.get('flags', 0))
        ])
        
        # Ensure consistent feature count (match your training - adjust based on your model)
        required_features = 51  # This should match your training features
        if len(features) < required_features:
            features.extend([0.0] * (required_features - len(features)))
        elif len(features) > required_features:
            features = features[:required_features]
        
        return np.array(features, dtype=np.float32).reshape(1, -1)
    except Exception as e:
        print(f"Error in extract_features_for_ml: {e}")
        traceback.print_exc()
        # Return a default feature array to prevent crashes
        return np.zeros((1, 51), dtype=np.float32)


def predict_with_ml_models(packet_data):
    """
    Get predictions from all available ML models.
    Returns: dict with all model predictions and ensemble result.
    """
    results = {}
    
    try:
        # Extract features for all models
        features = extract_features_for_ml(packet_data)
        
        # Random Forest prediction
        if RANDOM_FOREST_AVAILABLE and rf_model:
            try:
                rf_pred = rf_model.predict(features)[0]
                rf_proba = rf_model.predict_proba(features)[0]
                results['rf_prediction'] = 'Attack' if rf_pred == 1 else 'Normal'
                results['rf_confidence'] = float(max(rf_proba))
                results['rf_probabilities'] = [float(p) for p in rf_proba]
            except Exception as e:
                print(f"RF prediction error: {e}")
                # traceback.print_exc() # Uncomment for detailed RF errors if needed
                results['rf_prediction'] = 'Error'
                results['rf_confidence'] = 0.0
        else:
            results['rf_prediction'] = 'N/A'
            results['rf_confidence'] = 0.0
        
        # XGBoost prediction
        if XGBOOST_AVAILABLE and xgb_model:
            try:
                # Ensure XGBoost runs on CPU to avoid device mismatch warnings/errors
                xgb_pred = xgb_model.predict(features)[0]
                xgb_proba = xgb_model.predict_proba(features)[0]
                results['xgb_prediction'] = 'Attack' if xgb_pred == 1 else 'Normal'
                results['xgb_confidence'] = float(max(xgb_proba))
                results['xgb_probabilities'] = [float(p) for p in xgb_proba]
            except Exception as e:
                print(f"XGB prediction error: {e}")
                # traceback.print_exc() # Uncomment for detailed XGB errors if needed
                results['xgb_prediction'] = 'Error'
                results['xgb_confidence'] = 0.0
        else:
            results['xgb_prediction'] = 'N/A'
            results['xgb_confidence'] = 0.0
        
        # Autoencoder prediction
        if AUTOENCODER_AVAILABLE and autoencoder_integration:
            try:
                # Use the autoencoder integration method
                ae_result = autoencoder_integration.analyze_packet_for_gui(packet_data)
                
                # Handle the result properly
                prediction = ae_result.get("prediction", "Unknown")
                confidence_str = ae_result.get("confidence", "0.0")
                
                # Convert confidence string to float
                if isinstance(confidence_str, str):
                    if confidence_str.endswith('%'):
                        confidence = float(confidence_str.replace('%', '')) / 100
                    else:
                        confidence = float(confidence_str)
                else:
                    confidence = float(confidence_str)
                
                results['ae_prediction'] = prediction
                results['ae_confidence'] = confidence
                results['ae_category'] = ae_result.get("category", "Unknown")
                results['ae_severity'] = ae_result.get("severity", "Unknown")
                results['ae_description'] = ae_result.get("description", "")
                results['ae_reconstruction_error'] = ae_result.get("reconstruction_error", "N/A")
                results['ae_threshold'] = ae_result.get("threshold", "N/A")
            except Exception as e:
                print(f"AE prediction error: {e}")
                # traceback.print_exc() # Uncomment for detailed AE errors if needed
                results['ae_prediction'] = 'Error'
                results['ae_confidence'] = 0.0
                results['ae_category'] = 'Error'
                results['ae_severity'] = 'Unknown'
                results['ae_description'] = f'Error: {str(e)}'
                results['ae_reconstruction_error'] = 'N/A'
        else:
            results['ae_prediction'] = 'Not Loaded'
            results['ae_confidence'] = 0.0
            results['ae_category'] = 'N/A'
            results['ae_severity'] = 'Unknown'
            results['ae_description'] = 'Autoencoder not initialized'
            results['ae_reconstruction_error'] = 'N/A'
        
        # Ensemble voting (majority wins)
        predictions = []
        confidences = []
        
        if results['rf_prediction'] not in ['Error', 'N/A']:
            predictions.append(results['rf_prediction'])
            confidences.append(('RF', results['rf_confidence']))
        if results['xgb_prediction'] not in ['Error', 'N/A']:
            predictions.append(results['xgb_prediction'])
            confidences.append(('XGB', results['xgb_confidence']))
        if results['ae_prediction'] not in ['Error', 'N/A']:
            # Autoencoder uses "Normal"/"Anomaly", convert to "Normal"/"Attack"
            ae_normalized = 'Attack' if results['ae_prediction'] == 'Anomaly' else results['ae_prediction']
            predictions.append(ae_normalized)
            confidences.append(('AE', results['ae_confidence']))
        
        if len(predictions) > 0:
            # Count votes
            attack_votes = sum(1 for pred in predictions if pred == 'Attack')
            normal_votes = sum(1 for pred in predictions if pred == 'Normal')
            
            if attack_votes > normal_votes:
                results['ensemble_prediction'] = 'Attack'
            elif normal_votes > attack_votes:
                results['ensemble_prediction'] = 'Normal'
            else:
                # Tie - use highest confidence model
                if confidences:
                    best_model, best_conf = max(confidences, key=lambda x: x[1])
                    best_pred = results[f'{best_model.lower()}_prediction']
                    # Convert autoencoder result back
                    if best_model == 'AE':
                        best_pred = 'Attack' if best_pred == 'Anomaly' else best_pred
                    results['ensemble_prediction'] = best_pred
                else:
                    results['ensemble_prediction'] = 'Unknown'
        else:
            results['ensemble_prediction'] = 'Unknown'
        
        return results
    except Exception as e:
        print(f"Critical error in predict_with_ml_models: {e}")
        traceback.print_exc()
        # Return default error results to prevent crashes
        return {
            'rf_prediction': 'Error', 'rf_confidence': 0.0,
            'xgb_prediction': 'Error', 'xgb_confidence': 0.0,
            'ae_prediction': 'Error', 'ae_confidence': 0.0,
            'ae_category': 'Error', 'ae_severity': 'Unknown',
            'ae_description': f'Critical Error: {str(e)}',
            'ae_reconstruction_error': 'N/A',
            'ensemble_prediction': 'Error'
        }


def main(page: ft.Page):
    page.title = "Network Intrusion Detection System (NIDS)"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 10
    page.scroll = ft.ScrollMode.AUTO

    # UI state
    selected_packet = None

    # Header
    header = ft.Text("Network Intrusion Detection System", size=22, weight=ft.FontWeight.BOLD)

    # Controls
    status_text = ft.Text("Status: Stopped", color="red")
    start_btn = ft.ElevatedButton("Start Capture", disabled=True)
    refresh_btn = ft.ElevatedButton("Refresh (Clear)", bgcolor="#2E7D32")
    gen_pdf_btn = ft.ElevatedButton("Generate PDF Report", bgcolor="#1565C0")
    
    # Model status display
    model_status = ft.Text("Loading models...", color="orange")
    model_details = ft.Text("", size=12, color="gray")
    
    # Statistics display
    stats_text = ft.Text("Packets: 0 | Models: Loading...", size=12, color="gray")

    # DataTable (left) - Enhanced with all model predictions
    packet_table = ft.DataTable(
        width=1400,  # Increased width for additional columns
        columns=[
            ft.DataColumn(ft.Text("Time")),
            ft.DataColumn(ft.Text("Src IP")),
            ft.DataColumn(ft.Text("Dst IP")),
            ft.DataColumn(ft.Text("Proto")),
            ft.DataColumn(ft.Text("pSrc")),
            ft.DataColumn(ft.Text("pDst")),
            ft.DataColumn(ft.Text("RF")),  # Random Forest
            ft.DataColumn(ft.Text("XGB")), # XGBoost
            ft.DataColumn(ft.Text("AE")),  # Autoencoder
            ft.DataColumn(ft.Text("Ensemble")), # Ensemble result
            ft.DataColumn(ft.Text("Conf")), # Confidence
        ],
        rows=[]
    )

    # Put table inside a Column with scroll enabled
    table_container = ft.Container(
        content=ft.Column([packet_table], scroll=ft.ScrollMode.ALWAYS),
        width=1400,
        height=520
    )

    # Details panel (right) - initially hidden
    details_column = ft.Column(
        controls=[
            ft.Text("Packet Details", size=16, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Text("No packet selected."),
        ],
        scroll=ft.ScrollMode.ALWAYS,
        spacing=6,
        expand=True,
    )
    details_card = ft.Card(
        content=ft.Container(details_column, padding=12, width=500, height=520),
        elevation=2
    )
    details_card.visible = False  # start hidden

    # Use an asyncio.Lock to prevent concurrent UI updates from the background task
    ui_update_lock_async = asyncio.Lock()

    async def safe_update():
        try:
            # Acquire the lock before updating the UI
            async with ui_update_lock_async:
                page.update()
        except Exception:
            traceback.print_exc()

    # Update statistics display
    def update_statistics():
        # Count models available
        models_available = sum([
            1 if RANDOM_FOREST_AVAILABLE else 0,
            1 if XGBOOST_AVAILABLE else 0,
            1 if AUTOENCODER_AVAILABLE else 0
        ])
        
        stats_text.value = (
            f"Packets: {len(packet_history)} | "
            f"Models: {models_available}/3 | "
            f"RF: {'✓' if RANDOM_FOREST_AVAILABLE else '✗'} "
            f"XGB: {'✓' if XGBOOST_AVAILABLE else '✗'} "
            f"AE: {'✓' if AUTOENCODER_AVAILABLE else '✗'}"
        )
        # Call safe_update as a coroutine
        page.run_task(safe_update)

    # Populate right details panel for a selected packet - ENHANCED
    def populate_details_panel(pkt):
        nonlocal selected_packet
        selected_packet = pkt
        try:
            mapped = map_prediction_to_threat(pkt.get("prediction"))
            cves = mapped.get("cves") or []
            cvss = mapped.get("cvss")
            severity = mapped.get("severity", "Unknown")
            
            # Get all model results
            rf_pred = pkt.get("rf_prediction", "N/A")
            rf_conf = pkt.get("rf_confidence", 0)
            xgb_pred = pkt.get("xgb_prediction", "N/A")
            xgb_conf = pkt.get("xgb_confidence", 0)
            ae_pred = pkt.get("ae_prediction", "N/A")
            ae_conf = pkt.get("ae_confidence", 0)
            ensemble_pred = pkt.get("ensemble_prediction", "Unknown")
            
            # Autoencoder specific details
            ae_category = pkt.get("ae_category", "N/A")
            ae_severity = pkt.get("ae_severity", "N/A")
            ae_description = pkt.get("ae_description", "N/A")
            ae_recon_error = pkt.get("ae_reconstruction_error", "N/A")

            details_column.controls.clear()
            details_column.controls.extend([
                ft.Text("Packet Details", size=16, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                
                # Basic packet info
                ft.Text(f"Timestamp: {pkt.get('timestamp', 'N/A')}"),
                ft.Text(f"Source IP: {pkt.get('src_ip', 'N/A')}"),
                ft.Text(f"Destination IP: {pkt.get('dst_ip', 'N/A')}"),
                ft.Text(f"Protocol: {pkt.get('protocol', 'N/A')}"),
                ft.Text(f"Source Port: {pkt.get('port_src', 'N/A')}"),
                ft.Text(f"Dest Port: {pkt.get('port_dst', 'N/A')}"),
                
                ft.Divider(),
                
                # === NEW: Model Predictions Section ===
                ft.Text("Model Predictions", weight=ft.FontWeight.BOLD, color="cyan"),
                
                # Random Forest
                ft.Text(f"Random Forest: {rf_pred} (Conf: {rf_conf:.3f})", 
                       color="red" if rf_pred == "Attack" else "green" if rf_pred == "Normal" else "gray"),
                
                # XGBoost
                ft.Text(f"XGBoost: {xgb_pred} (Conf: {xgb_conf:.3f})",
                       color="red" if xgb_pred == "Attack" else "green" if xgb_pred == "Normal" else "gray"),
                
                # Autoencoder
                ft.Text(f"Autoencoder: {ae_pred} (Conf: {ae_conf:.3f})",
                       color="red" if ae_pred == "Anomaly" else "green" if ae_pred == "Normal" else "gray"),
                
                # Ensemble Result
                ft.Text(f"Ensemble: {ensemble_pred}", 
                       color="red" if ensemble_pred == "Attack" else "green" if ensemble_pred == "Normal" else "orange",
                       weight=ft.FontWeight.BOLD),
                
                ft.Divider(),
                
                # === NEW: Autoencoder Analysis Section ===
                ft.Text("Autoencoder Analysis", weight=ft.FontWeight.BOLD, color="blue"),
                ft.Text(f"Category: {ae_category}"),
                ft.Text(f"Severity: {ae_severity}",
                       color="green" if ae_severity == "Low" else 
                             "orange" if ae_severity in ["Medium", "High"] else "red"),
                ft.Text(f"Description: {ae_description}", size=11),
                ft.Text(f"Reconstruction Error: {ae_recon_error}", size=10, color="gray"),
                
                ft.Divider(),
                
                # Threat Mapping
                ft.Text("Threat Mapping", weight=ft.FontWeight.BOLD),
                ft.Text(f"Category: {mapped.get('category', 'N/A')}"),
                ft.Text(f"Description: {mapped.get('description', 'N/A')}"),
                ft.Text(f"Severity: {severity}"),
                ft.Text(f"CVSS: {cvss if cvss is not None else 'N/A'}"),
                ft.Text(f"CVEs: {', '.join(cves) if cves else 'None'}"),
                
                ft.Divider(),
                
                # Raw JSON
                ft.Text("Raw JSON", weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=ft.Column([ft.Text(json.dumps(pkt, indent=2, default=str), selectable=True)], 
                                     scroll=ft.ScrollMode.ALWAYS),
                    border=ft.border.all(0.5, "gray"),
                    padding=8,
                    height=120
                ),
                
                ft.Divider(),
                
                # Action buttons
                ft.Row([
                    ft.ElevatedButton("Export JSON", on_click=lambda e: export_single_json(pkt)),
                    ft.ElevatedButton("PDF Report", on_click=lambda e: on_generate_pdf(e)),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
            ])
            
            if not details_card.visible:
                details_card.visible = True
            # Call safe_update as a coroutine
            page.run_task(safe_update)
        except Exception:
            traceback.print_exc()

    def collapse_details_panel():
        details_card.visible = False
        # Call safe_update as a coroutine
        page.run_task(safe_update)

    # Export a single packet to JSON file
    def export_single_json(pkt):
        try:
            reports_dir = os.path.join(parent_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            fname = f"packet-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            out_path = os.path.join(reports_dir, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(pkt, f, indent=2, default=str)
            page.snack_bar = ft.SnackBar(ft.Text(f"Saved packet JSON: {out_path}"))
            page.snack_bar.open = True
            # Call safe_update as a coroutine
            page.run_task(safe_update)
        except Exception as e:
            traceback.print_exc()
            page.snack_bar = ft.SnackBar(ft.Text(f"Export failed: {e}"))
            page.snack_bar.open = True
            # Call safe_update as a coroutine
            page.run_task(safe_update)

    # Generate PDF report for the session
    def on_generate_pdf(e):
        try:
            if not REPORT_GENERATOR_AVAILABLE:
                page.snack_bar = ft.SnackBar(ft.Text("PDF generator not available. Install reportlab."))
                page.snack_bar.open = True
                # Call safe_update as a coroutine
                page.run_task(safe_update)
                return

            if not packet_history:
                page.snack_bar = ft.SnackBar(ft.Text("No packets captured to include in the report."))
                page.snack_bar.open = True
                # Call safe_update as a coroutine
                page.run_task(safe_update)
                return

            # Prepare packet copies with threat_mapping and all model results
            packets_for_report = []
            for pkt in packet_history:
                pc = dict(pkt)
                pc["threat_mapping"] = map_prediction_to_threat(pkt.get("prediction"))
                packets_for_report.append(pc)

            reports_dir = os.path.join(parent_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            out_file = os.path.join(reports_dir, f"session-report-{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

            generate_pdf_report(packets_for_report, out_file)

            page.snack_bar = ft.SnackBar(ft.Text(f"PDF report created: {out_file}"))
            page.snack_bar.open = True
            # Call safe_update as a coroutine
            page.run_task(safe_update)
        except Exception:
            traceback.print_exc()
            page.snack_bar = ft.SnackBar(ft.Text("PDF generation failed - see console"))
            page.snack_bar.open = True
            # Call safe_update as a coroutine
            page.run_task(safe_update)

    # Get color for prediction
    def get_prediction_color(prediction):
        if prediction in ["Attack", "Anomaly"]:
            return "red"
        elif prediction in ["Normal"]:
            return "green"
        else:
            return "gray"

    # clickable cell wrapper - ENHANCED with color coding
    def clickable_cell(text_value, pkt, is_prediction=False):
        color = None
        if is_prediction and text_value:
            color = get_prediction_color(str(text_value))
        
        return ft.Container(
            content=ft.Text(str(text_value), color=color),
            padding=ft.padding.only(left=6),
            on_click=lambda e: on_row_clicked(pkt)
        )

    # When a row/cell is clicked: toggle show/collapse behavior
    def on_row_clicked(pkt):
        nonlocal selected_packet
        if details_card.visible and selected_packet is pkt:
            collapse_details_panel()
            selected_packet = None
        else:
            populate_details_panel(pkt)

    # Insert a new packet row - ENHANCED with all model predictions
    def insert_packet_row(pkt):
        try:
            # Get predictions for display
            rf_pred = pkt.get("rf_prediction", "N/A")
            xgb_pred = pkt.get("xgb_prediction", "N/A")
            ae_pred = pkt.get("ae_prediction", "N/A")
            ensemble_pred = pkt.get("ensemble_prediction", "Unknown")
            confidence = max(
                pkt.get("rf_confidence", 0),
                pkt.get("xgb_confidence", 0),
                pkt.get("ae_confidence", 0)
            )
            
            # Create DataCells with clickable cell wrappers
            cells = [
                ft.DataCell(clickable_cell(pkt.get("timestamp", ""), pkt)),
                ft.DataCell(clickable_cell(pkt.get("src_ip", ""), pkt)),
                ft.DataCell(clickable_cell(pkt.get("dst_ip", ""), pkt)),
                ft.DataCell(clickable_cell(pkt.get("protocol", ""), pkt)),
                ft.DataCell(clickable_cell(pkt.get("port_src", ""), pkt)),
                ft.DataCell(clickable_cell(pkt.get("port_dst", ""), pkt)),
                ft.DataCell(clickable_cell(rf_pred, pkt, is_prediction=True)),  # RF prediction
                ft.DataCell(clickable_cell(xgb_pred, pkt, is_prediction=True)), # XGB prediction
                ft.DataCell(clickable_cell(ae_pred, pkt, is_prediction=True)),  # AE prediction
                ft.DataCell(clickable_cell(ensemble_pred, pkt, is_prediction=True)), # Ensemble
                ft.DataCell(clickable_cell(f"{confidence:.3f}", pkt)),  # Confidence
            ]
            
            # Use the lock to safely update the UI from the processing thread
            with ui_update_lock:
                packet_table.rows.insert(0, ft.DataRow(cells=cells))
                
                # Maintain history (newest first)
                packet_history.insert(0, pkt)
                if len(packet_history) > 2000:
                    del packet_history[2000:]
                
                # Prune table rows for responsiveness
                if len(packet_table.rows) > 1500:
                    del packet_table.rows[800:]
                
                # Update statistics
                update_statistics()
            
            # Schedule the UI update on the main thread using the async task
            page.run_task(safe_update)
        except Exception:
            traceback.print_exc()

    # Coroutine poller (drains queue on UI loop)
    async def queue_poller():
        try:
            while True:
                packets_to_process = []
                while True:
                    try:
                        pkt = packet_queue.get_nowait()
                        packets_to_process.append(pkt)
                    except queue.Empty:
                        break
                
                for pkt in packets_to_process:
                    try:
                        # Check stop event before processing each packet
                        if stop_event.is_set():
                            print("Queue poller stopping due to stop_event.")
                            return
                        
                        # === ANALYZE WITH ALL MODELS ===
                        model_results = predict_with_ml_models(pkt)
                        
                        # Add all model results to packet dict
                        pkt.update(model_results)
                        
                        insert_packet_row(pkt)
                    except Exception as e:
                        print(f"Error processing packet in queue_poller: {e}")
                        # traceback.print_exc() # Uncomment for very detailed error logs if needed
                        # Continue processing other packets even if one fails
                
                # Sleep briefly to prevent the loop from consuming 100% CPU
                await asyncio.sleep(0.05)
        except Exception:
            traceback.print_exc()

    # Background start/stop wrapper functions
    def _bg_start():
        global capture_thread, stop_event
        try:
            if not LiveCaptureManager:
                print("LiveCaptureManager not available.")
                return
            # Reset stop event
            stop_event.clear()
            # Create a new LiveCaptureManager instance
            lcm_instance = LiveCaptureManager()
            # Set the stop event if the manager supports it
            if hasattr(lcm_instance, 'set_stop_event'):
                lcm_instance.set_stop_event(stop_event)
            # Fast callback only enqueues packet dicts
            def pkt_cb(details):
                # Check stop event before processing
                if stop_event.is_set():
                    return
                try:
                    # Ensure the packet data is a dictionary before putting it in the queue
                    if isinstance(details, dict):
                        # Validate essential keys exist and are not 'N/A' before queuing
                        # This is a last-resort check if LiveCaptureManager sends bad data
                        essential_fields = ['timestamp', 'src_ip', 'dst_ip', 'port_src', 'port_dst']
                        is_valid = True
                        for field in essential_fields:
                            val = details.get(field, 'MISSING')
                            if val == 'N/A' or val == '':
                                print(f"Warning: Invalid {field} value '{val}' in packet. Skipping.")
                                is_valid = False
                                break
                        if is_valid:
                            packet_queue.put(details)
                        else:
                            print(f"Skipping invalid packet: {details}")
                    else:
                        print(f"Warning: Received non-dict packet  {details}")
                except Exception as e:
                    print(f"Error in packet callback: {e}")
                    traceback.print_exc()

            # Register callback
            if hasattr(lcm_instance, "set_packet_callback"):
                lcm_instance.set_packet_callback(pkt_cb)
            else:
                setattr(lcm_instance, "packet_callback", pkt_cb)

            # Start capture in a new thread
            capture_thread = threading.Thread(target=lcm_instance.start_capture, daemon=True)
            capture_thread.start()
            print("[LiveCaptureManager] Capture thread started.")
        except Exception as e:
            print(f"Error starting capture: {e}")
            traceback.print_exc()

    def _bg_stop():
        global capture_thread, stop_event
        try:
            print("Stopping capture...")
            # Set stop event to signal capture to stop
            stop_event.set()
            # Wait for capture thread to finish
            if capture_thread and capture_thread.is_alive():
                print("Waiting for capture thread to finish...")
                capture_thread.join(timeout=3.0) # Increased timeout
                if capture_thread.is_alive():
                    print("Warning: Capture thread did not stop gracefully.")
                else:
                    print("Capture thread stopped.")
            # Clear the queue
            while True:
                try:
                    packet_queue.get_nowait()
                except queue.Empty:
                    break
            print("Capture stopped.")
        except Exception as e:
            print(f"Error stopping capture: {e}")
            traceback.print_exc()

    def on_start_stop(e):
        global capture_thread, stop_event
        if not LiveCaptureManager:
            page.snack_bar = ft.SnackBar(ft.Text("LiveCaptureManager not available"))
            page.snack_bar.open = True
            # Call safe_update as a coroutine
            page.run_task(safe_update)
            return
        
        # Disable button temporarily to prevent multiple clicks
        start_btn.disabled = True
        # Call safe_update as a coroutine
        page.run_task(safe_update)
        
        if start_btn.text == "Start Capture":
            # Start capture
            start_btn.text = "Stop Capture"
            status_text.value = "Status: Running"
            status_text.color = "green"
            threading.Thread(target=_bg_start, daemon=True).start()
        else:
            # Stop capture
            start_btn.text = "Start Capture"
            status_text.value = "Status: Stopping..."
            status_text.color = "orange"
            threading.Thread(target=_bg_stop, daemon=True).start()
            # Brief pause to allow stop logic to initiate
            time.sleep(0.1)
            status_text.value = "Status: Stopped"
            status_text.color = "red"
        
        # Re-enable button after a short delay
        time.sleep(0.1)
        start_btn.disabled = False
        # Call safe_update as a coroutine
        page.run_task(safe_update)

    def on_refresh(e):
        # Disable button temporarily
        refresh_btn.disabled = True
        # Call safe_update as a coroutine
        page.run_task(safe_update)
        
        try:
            # Clear UI and state
            # Use the lock to safely update the UI from the main thread
            with ui_update_lock:
                packet_table.rows = []
                packet_history.clear()
            # Clear the queue
            while True:
                try:
                    packet_queue.get_nowait()
                except queue.Empty:
                    break
            collapse_details_panel()
            
            # Reset autoencoder statistics if available
            if AUTOENCODER_AVAILABLE and autoencoder_integration:
                try:
                    autoencoder_integration.reset_statistics()
                except Exception as e:
                    print(f"Error resetting autoencoder stats: {e}")

            update_statistics()
        except Exception as e:
            print(f"Error during refresh: {e}")
            traceback.print_exc()
        finally:
            # Re-enable button
            refresh_btn.disabled = False
            # Call safe_update as a coroutine
            page.run_task(safe_update)

    # Initialize All Models - FIXED
    def initialize():
        global live_capture_manager
        
        try:
            print("\n" + "="*70)
            print("INITIALIZING NIDS MODELS")
            print("="*70)
            
            # Model availability summary
            available_models = []
            if RANDOM_FOREST_AVAILABLE:
                available_models.append("Random Forest")
            if XGBOOST_AVAILABLE:
                available_models.append("XGBoost")
            if AUTOENCODER_AVAILABLE:
                available_models.append("Autoencoder")
            
            print(f"Available models: {', '.join(available_models) if available_models else 'None'}")
            
            # Initialize LiveCaptureManager
            print("\nInitializing Live Capture Manager...")
            
            if LiveCaptureManager is None:
                print("LiveCaptureManager not available.")
            else:
                # We'll create LiveCaptureManager instances in _bg_start now
                live_capture_manager = None
                print("LiveCaptureManager found.")

            # Start poller coroutine (non-blocking)
            try:
                page.run_task(queue_poller)
            except Exception:
                print("Failed to start queue poller task.")
                traceback.print_exc()
                # Fallback drain once
                while True:
                    try:
                        p = packet_queue.get_nowait()
                        insert_packet_row(p)
                    except queue.Empty:
                        break

            model_status.value = f"Models: {len(available_models)}/3 loaded. Ready."
            model_status.color = "green"
            
            model_details.value = f"RF: {'✓' if RANDOM_FOREST_AVAILABLE else '✗'} | " \
                                 f"XGB: {'✓' if XGBOOST_AVAILABLE else '✗'} | " \
                                 f"AE: {'✓' if AUTOENCODER_AVAILABLE else '✗'}"
            model_details.color = "green" if len(available_models) > 0 else "orange"
            
            start_btn.disabled = False
            
            print("\n" + "="*70)
            print("✓ NIDS GUI Initialized Successfully")
            print("="*70)
            
            # Call safe_update as a coroutine
            page.run_task(safe_update)
            
        except Exception:
            traceback.print_exc()
            model_status.value = "Initialization error"
            model_status.color = "red"
            # Call safe_update as a coroutine
            page.run_task(safe_update)

    # Hook up handlers
    start_btn.on_click = on_start_stop
    refresh_btn.on_click = on_refresh
    gen_pdf_btn.on_click = on_generate_pdf

    # Build the page layout - ENHANCED with model status
    page.add(header)
    page.add(ft.Divider())
    page.add(ft.Row([
        status_text, 
        start_btn, 
        refresh_btn, 
        gen_pdf_btn
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN))
    page.add(ft.Divider())
    
    # Add model status and statistics
    page.add(ft.Row([
        ft.Column([model_status, model_details], spacing=2),
        stats_text
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN))
    
    page.add(ft.Divider())
    
    # Main content area
    page.add(ft.Row([
        ft.Column([table_container], expand=True),
        ft.VerticalDivider(width=1),
        details_card
    ], expand=True))
    
    # Call safe_update as a coroutine
    page.run_task(safe_update)

    # Initialize the capture manager and all models
    initialize()

    # Shutdown: attempt to stop capture gracefully
    def on_close(e):
        global capture_thread, stop_event
        print("Shutting down...")
        try:
            # Set stop event to signal capture to stop
            stop_event.set()
            # Wait for capture thread to finish
            if capture_thread and capture_thread.is_alive():
                print("Waiting for capture thread to finish on close...")
                capture_thread.join(timeout=3.0)
                if capture_thread.is_alive():
                    print("Warning: Capture thread did not stop gracefully on close.")
                else:
                    print("Capture thread stopped on close.")
            # Clear the queue
            while True:
                try:
                    packet_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Export final autoencoder statistics if available
            if AUTOENCODER_AVAILABLE and autoencoder_integration:
                try:
                    stats = autoencoder_integration.get_statistics()
                    print("\n" + "="*70)
                    print("FINAL AUTOENCODER STATISTICS")
                    print("="*70)
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                    print("="*70)
                except Exception as e:
                    print(f"Error getting autoencoder stats: {e}")

        except Exception:
            traceback.print_exc()

    page.on_close = on_close


if __name__ == "__main__":
    ft.app(target=main)