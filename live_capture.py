# src/live_capture.py
"""
LiveCaptureManager (safe pause/resume approach)

- Uses Scapy AsyncSniffer when available.
- Starts sniffer once; uses _accept_packets flag to pause/resume delivering packets.
- Provides shutdown() to fully stop the sniffer when app exits.
- packet_callback must be a callable that accepts one dict argument (we only enqueue in callback).
"""

import os
import json
import joblib
import threading
import time
import traceback
import pandas as pd

# Try scapy AsyncSniffer
try:
    from scapy.all import AsyncSniffer, IP, IPv6, TCP, UDP  # noqa: F401
    SCAPY_AVAILABLE = True
except Exception:
    SCAPY_AVAILABLE = False

class LiveCaptureManager:
    def __init__(
        self,
        rf_path="models/trained_randomforest_sampled_smote.pkl",
        xgb_path="models/trained_xgboost_sampled_classweights.pkl",
        features_path="models/feature_names.json",
        missing_value=0,
    ):
        self.rf_path = rf_path
        self.xgb_path = xgb_path
        self.features_path = features_path
        self.missing_value = missing_value

        # models & feature names
        self.rf_model = None
        self.xgb_model = None
        self.rf_feature_names = None
        self.xgb_feature_names = None

        # callback and control
        self.packet_callback = None  # GUI registers this (must be thread-safe)
        self._sniffer = None
        self._sim_thread = None
        self._sim_stop_event = threading.Event()

        # active vs accepting packets:
        # active: user-visible start/stop state (True when user clicked Start)
        # _sniffer_running: whether async sniffer is running (may be True even when active False)
        # _accept_packets: whether we should deliver packets to callback (True when active)
        self.active = False
        self._sniffer_running = False
        self._accept_packets = False

        # load models/features
        self._load_models_and_features()

    # -------------------------
    # Model & feature utilities
    # -------------------------
    def _load_models_and_features(self):
        try:
            if os.path.exists(self.rf_path):
                self.rf_model = joblib.load(self.rf_path)
            if os.path.exists(self.xgb_path):
                self.xgb_model = joblib.load(self.xgb_path)

            self.rf_feature_names = self._extract_feature_names_from_model(self.rf_model) if self.rf_model else None
            self.xgb_feature_names = self._extract_feature_names_from_model(self.xgb_model) if self.xgb_model else None

            if self.rf_feature_names is None or self.xgb_feature_names is None:
                self._load_features_from_json(self.features_path)

            if self.rf_feature_names is None and self.xgb_feature_names is not None:
                self.rf_feature_names = list(self.xgb_feature_names)
            if self.xgb_feature_names is None and self.rf_feature_names is not None:
                self.xgb_feature_names = list(self.rf_feature_names)
        except Exception as e:
            print("[LiveCaptureManager] model/features load error:", e)
            traceback.print_exc()

    def _extract_feature_names_from_model(self, model):
        if model is None:
            return None
        try:
            if hasattr(model, "feature_names_in_"):
                return list(model.feature_names_in_)
            for attr in ("feature_names", "feature_list", "features", "columns"):
                if hasattr(model, attr):
                    try:
                        return list(getattr(model, attr))
                    except Exception:
                        pass
            try:
                if hasattr(model, "get_booster"):
                    booster = model.get_booster()
                    if hasattr(booster, "feature_names") and booster.feature_names:
                        return list(booster.feature_names)
                if hasattr(model, "booster_"):
                    booster = model.booster_
                    if hasattr(booster, "feature_names") and booster.feature_names:
                        return list(booster.feature_names)
            except Exception:
                pass
            if hasattr(model, "named_steps"):
                for name, step in model.named_steps.items():
                    if hasattr(step, "feature_names_in_"):
                        return list(step.feature_names_in_)
        except Exception:
            pass
        return None

    def _load_features_from_json(self, path):
        try:
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "rf" in data:
                    self.rf_feature_names = list(data["rf"])
                if "xgb" in data:
                    self.xgb_feature_names = list(data["xgb"])
                if "features" in data:
                    if self.rf_feature_names is None:
                        self.rf_feature_names = list(data["features"])
                    if self.xgb_feature_names is None:
                        self.xgb_feature_names = list(data["features"])
            elif isinstance(data, list):
                if self.rf_feature_names is None:
                    self.rf_feature_names = list(data)
                if self.xgb_feature_names is None:
                    self.xgb_feature_names = list(data)
        except Exception as e:
            print("[LiveCaptureManager] failed loading features json:", e)
            traceback.print_exc()

    # -------------------------
    # Public API
    # -------------------------
    def set_packet_callback(self, callback):
        if not callable(callback):
            raise ValueError("callback must be callable")
        self.packet_callback = callback

    # -------------------------
    # Build input & predict
    # -------------------------
    def _build_feature_input(self, pkt_info: dict, model="rf"):
        fnames = self.rf_feature_names if model == "rf" else self.xgb_feature_names
        if fnames is None:
            keys = sorted(pkt_info.keys())
            vals = []
            for k in keys:
                try:
                    vals.append(float(pkt_info.get(k, self.missing_value)))
                except Exception:
                    vals.append(float(self.missing_value))
            return pd.DataFrame([vals], columns=keys)
        row = {}
        for fn in fnames:
            if fn in pkt_info:
                val = pkt_info[fn]
            else:
                found = False
                for k in pkt_info:
                    if k.lower() == fn.lower():
                        val = pkt_info[k]
                        found = True
                        break
                if not found:
                    val = self.missing_value
            try:
                row[fn] = float(val)
            except Exception:
                row[fn] = float(self.missing_value)
        return pd.DataFrame([row], columns=fnames)

    def predict_packet(self, pkt_info: dict, return_proba=False):
        results = {"rf": None, "xgb": None}
        if self.rf_model is not None:
            try:
                Xrf = self._build_feature_input(pkt_info, model="rf")
                lbl = self.rf_model.predict(Xrf)[0]
                proba = None
                if return_proba and hasattr(self.rf_model, "predict_proba"):
                    try:
                        proba = self.rf_model.predict_proba(Xrf)[0].tolist()
                    except Exception:
                        proba = None
                results["rf"] = {"label": int(lbl), "proba": proba}
            except Exception:
                traceback.print_exc()
        if self.xgb_model is not None:
            try:
                Xx = self._build_feature_input(pkt_info, model="xgb")
                lbl = self.xgb_model.predict(Xx)[0]
                proba = None
                if return_proba and hasattr(self.xgb_model, "predict_proba"):
                    try:
                        proba = self.xgb_model.predict_proba(Xx)[0].tolist()
                    except Exception:
                        proba = None
                results["xgb"] = {"label": int(lbl), "proba": proba}
            except Exception:
                traceback.print_exc()
        return results

    # -------------------------
    # Packet handling (scapy)
    # -------------------------
    def _packet_handler(self, pkt):
        # fast return if we're paused (do not process nor call callback)
        if not self._accept_packets:
            return
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            pkt_base = {
                "timestamp": ts,
                "src_ip": None,
                "dst_ip": None,
                "protocol": "OTHER",
                "port_src": None,
                "port_dst": None,
            }

            try:
                if pkt.haslayer("IP"):
                    ip = pkt.getlayer("IP")
                    pkt_base["src_ip"] = getattr(ip, "src", None)
                    pkt_base["dst_ip"] = getattr(ip, "dst", None)
                elif pkt.haslayer("IPv6"):
                    ip6 = pkt.getlayer("IPv6")
                    pkt_base["src_ip"] = getattr(ip6, "src", None)
                    pkt_base["dst_ip"] = getattr(ip6, "dst", None)
            except Exception:
                pass

            try:
                if pkt.haslayer("TCP"):
                    t = pkt.getlayer("TCP")
                    pkt_base["protocol"] = "TCP"
                    pkt_base["port_src"] = int(getattr(t, "sport", 0) or 0)
                    pkt_base["port_dst"] = int(getattr(t, "dport", 0) or 0)
                    try:
                        pkt_base["payload_len"] = len(bytes(t.payload))
                    except Exception:
                        pkt_base["payload_len"] = 0
                elif pkt.haslayer("UDP"):
                    u = pkt.getlayer("UDP")
                    pkt_base["protocol"] = "UDP"
                    pkt_base["port_src"] = int(getattr(u, "sport", 0) or 0)
                    pkt_base["port_dst"] = int(getattr(u, "dport", 0) or 0)
                    try:
                        pkt_base["payload_len"] = len(bytes(u.payload))
                    except Exception:
                        pkt_base["payload_len"] = 0
                else:
                    try:
                        pkt_base["payload_len"] = len(bytes(pkt.payload))
                    except Exception:
                        pkt_base["payload_len"] = 0
            except Exception:
                pass

            # build pkt_features according to feature names if known
            pkt_features = {}
            if self.rf_feature_names is not None:
                for fn in self.rf_feature_names:
                    if fn in pkt_base:
                        pkt_features[fn] = pkt_base[fn]
                    else:
                        matched = None
                        for k in pkt_base:
                            if k.lower() == fn.lower():
                                matched = pkt_base[k]; break
                        pkt_features[fn] = matched if matched is not None else self.missing_value
            else:
                pkt_features = dict(pkt_base)

            # predictions
            try:
                preds = self.predict_packet(pkt_features, return_proba=True)
            except Exception:
                preds = {"rf": None, "xgb": None}

            label_text = "Unknown"
            confidence_text = "N/A"
            model_used = "None"
            chosen = None
            if preds.get("rf") is not None:
                chosen = preds["rf"]; model_used = "RF"
            elif preds.get("xgb") is not None:
                chosen = preds["xgb"]; model_used = "XGB"
            if chosen is not None:
                try:
                    lbl = int(chosen.get("label", 0))
                    label_text = "Attack" if lbl != 0 else "Benign"
                    proba = chosen.get("proba")
                    if proba and isinstance(proba, (list, tuple)):
                        confidence_text = f"{max(proba) * 100:.1f}%"
                    else:
                        confidence_text = str(proba) if proba is not None else "N/A"
                except Exception:
                    label_text = "Unknown"

            details = {
                "timestamp": pkt_base.get("timestamp"),
                "src_ip": pkt_base.get("src_ip"),
                "dst_ip": pkt_base.get("dst_ip"),
                "protocol": pkt_base.get("protocol"),
                "port_src": pkt_base.get("port_src"),
                "port_dst": pkt_base.get("port_dst"),
                "rf": preds.get("rf"),
                "xgb": preds.get("xgb"),
                "prediction": label_text,
                "confidence": confidence_text,
                "model_used": model_used,
            }

            # safe callback invocation: do not block; catch exceptions
            try:
                if callable(self.packet_callback):
                    # we only call callback (which is expected to be fast and thread-safe)
                    self.packet_callback(details)
            except Exception:
                traceback.print_exc()

        except Exception:
            traceback.print_exc()

    # -------------------------
    # Start / Stop / Shutdown
    # -------------------------
    def start_capture(self, iface=None):
        """
        Start capture: ensure sniffer running and mark accepting packets = True.
        Starting while sniffer already running simply sets _accept_packets True.
        """
        try:
            # start sniffer once (if scapy available)
            if SCAPY_AVAILABLE and not self._sniffer_running:
                try:
                    self._sniffer = AsyncSniffer(iface=iface, prn=self._packet_handler, store=False)
                    self._sniffer.start()
                    self._sniffer_running = True
                except Exception:
                    traceback.print_exc()
                    self._sniffer = None
                    self._sniffer_running = False

            # fallback to simulated thread if scapy not available and sniffer not running
            if not SCAPY_AVAILABLE and (self._sim_thread is None or not self._sim_thread.is_alive()):
                self._sim_stop_event.clear()
                self._sim_thread = threading.Thread(target=self._simulated_loop, args=(iface,), daemon=True)
                self._sim_thread.start()

            # mark active & accept packets quickly (non-blocking)
            self.active = True
            self._accept_packets = True
            print("[LiveCaptureManager] start_capture -> active True, _accept_packets True")
        except Exception:
            traceback.print_exc()

    def stop_capture(self):
        """
        Stop capturing from GUI perspective: set _accept_packets False and active False.
        We do not call AsyncSniffer.stop() here to avoid blocking/race conditions.
        For full shutdown call shutdown().
        """
        try:
            self._accept_packets = False
            self.active = False
            print("[LiveCaptureManager] stop_capture -> _accept_packets False, active False")
        except Exception:
            traceback.print_exc()

    def shutdown(self, wait_timeout=1.0):
        """
        Full shutdown (call on app exit). Attempts to stop AsyncSniffer cleanly
        and joins simulated thread if used.
        """
        try:
            # ensure we stop accepting packets immediately
            self._accept_packets = False
            self.active = False

            if self._sniffer is not None:
                try:
                    self._sniffer.stop()
                except Exception:
                    traceback.print_exc()
                finally:
                    self._sniffer = None
                    self._sniffer_running = False

            if self._sim_thread is not None:
                try:
                    self._sim_stop_event.set()
                    self._sim_thread.join(timeout=wait_timeout)
                except Exception:
                    traceback.print_exc()
                finally:
                    self._sim_thread = None
                    self._sim_stop_event.clear()

            print("[LiveCaptureManager] shutdown completed.")
        except Exception:
            traceback.print_exc()

    # -------------------------
    # Simulated loop (fallback)
    # -------------------------
    def _simulated_loop(self, iface):
        try:
            while True:
                if self._sim_stop_event.is_set():
                    break
                if self._accept_packets:
                    pkt_base = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "src_ip": "192.168.1.10",
                        "dst_ip": "192.168.1.100",
                        "protocol": "TCP",
                        "port_src": 12345,
                        "port_dst": 80,
                    }
                    pkt_features = {}
                    if self.rf_feature_names is not None:
                        for fn in self.rf_feature_names:
                            if fn in pkt_base:
                                pkt_features[fn] = pkt_base[fn]
                            else:
                                matched = None
                                for k in pkt_base:
                                    if k.lower() == fn.lower():
                                        matched = pkt_base[k]; break
                                pkt_features[fn] = matched if matched is not None else self.missing_value
                    else:
                        pkt_features = dict(pkt_base)
                    try:
                        preds = self.predict_packet(pkt_features, return_proba=True)
                    except Exception:
                        preds = {"rf": None, "xgb": None}
                    label_text = "Unknown"; confidence_text = "N/A"; model_used = "None"; chosen = None
                    if preds.get("rf") is not None:
                        chosen = preds["rf"]; model_used = "RF"
                    elif preds.get("xgb") is not None:
                        chosen = preds["xgb"]; model_used = "XGB"
                    if chosen is not None:
                        try:
                            lbl = int(chosen.get("label", 0))
                            label_text = "Attack" if lbl != 0 else "Benign"
                            proba = chosen.get("proba")
                            if proba and isinstance(proba, (list, tuple)):
                                confidence_text = f"{max(proba) * 100:.1f}%"
                            else:
                                confidence_text = str(proba) if proba is not None else "N/A"
                        except Exception:
                            label_text = "Unknown"
                    details = {
                        "timestamp": pkt_base.get("timestamp"),
                        "src_ip": pkt_base.get("src_ip"),
                        "dst_ip": pkt_base.get("dst_ip"),
                        "protocol": pkt_base.get("protocol"),
                        "port_src": pkt_base.get("port_src"),
                        "port_dst": pkt_base.get("port_dst"),
                        "rf": preds.get("rf"),
                        "xgb": preds.get("xgb"),
                        "prediction": label_text,
                        "confidence": confidence_text,
                        "model_used": model_used,
                    }
                    # deliver to callback (safe)
                    try:
                        if callable(self.packet_callback):
                            self.packet_callback(details)
                    except Exception:
                        traceback.print_exc()
                time.sleep(0.6)
        except Exception:
            traceback.print_exc()

    # -------------------------
    # test emit
    # -------------------------
    def test_emit(self):
        pkt = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "protocol": "ICMP",
            "port_src": 0,
            "port_dst": 0,
            "prediction": "Benign",
            "confidence": "100%",
            "model_used": "TEST",
        }
        try:
            if callable(self.packet_callback):
                self.packet_callback(pkt)
        except Exception:
            traceback.print_exc()
