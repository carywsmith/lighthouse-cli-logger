#!/usr/bin/env python3
# Lighthouse ApexRp / Remote P Modbus TCP Logger
# - Strictly Modbus/TCP
# - Auto-detect Apex vs Remote P
# - Optional InfluxDB (toggle with --no-influx)
# - Auto-start sampling and set Sample/Hold/Delay (seconds or minutes)
# - Calculates per-channel concentration as particles / 0.1 ft^3
# - Optional Temperature/RH from user-specified input registers

import argparse
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any

from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

# InfluxDB (optional)
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    HAVE_INFLUX = True
except Exception:
    HAVE_INFLUX = False
    InfluxDBClient = None
    Point = None
    WritePrecision = None

# -------- Helpers for Modbus address math --------
def h_addr(reg: int) -> int:
    if reg < 40001:
        raise ValueError(f"Holding register must be >= 40001, got {reg}")
    return reg - 40001

def i_addr(reg: int) -> int:
    if reg < 30001:
        raise ValueError(f"Input register must be >= 30001, got {reg}")
    return reg - 30001

def regs_to_u32(high: int, low: int) -> int:
    return ((high & 0xFFFF) << 16) | (low & 0xFFFF)

def regs_to_i32(high: int, low: int) -> int:
    val = regs_to_u32(high, low)
    return struct.unpack('>i', struct.pack('>I', val))[0]

def regs_to_float_ieee(high: int, low: int, word_order_big_endian: bool = True) -> float:
    # IEEE754 float from two 16-bit regs; default word order is High then Low (big-endian words)
    if word_order_big_endian:
        raw = struct.pack('>HH', high & 0xFFFF, low & 0xFFFF)
    else:
        raw = struct.pack('>HH', low & 0xFFFF, high & 0xFFFF)
    return struct.unpack('>f', raw)[0]

def decode_ascii_words(words: List[int]) -> str:
    bs = bytearray()
    for w in words:
        bs.append((w >> 8) & 0xFF); bs.append(w & 0xFF)
    if 0 in bs: bs = bs[:bs.index(0)]
    try:
        return bs.decode('ascii', errors='ignore').strip()
    except Exception:
        return ''

# -------- Device register maps (only what we use) --------
@dataclass
class DeviceProfile:
    name: str
    type: str  # 'apex' or 'remote'
    FLOW_REG: int = 40023                 # /100 => CFM (per manuals)
    DEVICE_STATUS: int = 40003
    EXT_STATUS: Optional[int] = None      # Apex uses 40056
    # Apex runtime alarm masks (Remote P uses Sample Status instead)
    ALARM_PARTICLE_HI: Optional[int] = None
    ALARM_PARTICLE_LO: Optional[int] = None
    ALARM_ANALOG_HI: Optional[int] = None
    ALARM_ANALOG_LO: Optional[int] = None
    PRODUCT_NAME_START: int = 40007
    PRODUCT_NAME_LEN: int = 8
    MODEL_NAME_START: int = 40015
    MODEL_NAME_LEN: int = 8

APEX_PROFILE = DeviceProfile(
    name="ApexRp",
    type="apex",
    EXT_STATUS=40056,
    ALARM_PARTICLE_HI=40064,
    ALARM_PARTICLE_LO=40065,
    ALARM_ANALOG_HI=40066,
    ALARM_ANALOG_LO=40067,
)
REMOTE_PROFILE = DeviceProfile(
    name="REMOTE P",
    type="remote",
    EXT_STATUS=None,
)

# -------- Core class --------
class LighthouseLogger:
    def __init__(
        self,
        host: str,
        port: int,
        unit_id: int,
        influx_url: str,
        influx_org: str,
        influx_bucket: str,
        influx_token: str,
        interval: float = 5.0,
        max_channels: int = 6,
        verbose: bool = False,
        log_file: Optional[str] = None,
        enable_influx: bool = True,
    ):
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.interval = interval
        self.max_channels = max_channels

        self.client = ModbusTcpClient(host=host, port=port, timeout=3)
        self.profile: Optional[DeviceProfile] = None
        self.product_name = ''
        self.model_name = ''
        self.register_map_version = None

        self._influx_bucket = influx_bucket
        self._influx_org = influx_org
        self.enable_influx = bool(enable_influx) and HAVE_INFLUX and bool(influx_token)
        if self.enable_influx:
            self.influx = InfluxDBClient(url=influx_url, token=influx_token, org=influx_org)
            self.write_api = self.influx.write_api()
        else:
            self.influx = None
            self.write_api = None

        # logging
        self.logger = logging.getLogger("lighthouse")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(logging.DEBUG if verbose else logging.INFO)
        self.logger.addHandler(sh)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            fh.setLevel(logging.DEBUG if verbose else logging.INFO)
            self.logger.addHandler(fh)
        if not self.enable_influx:
            if enable_influx is False:
                why = "disabled by --no-influx"
            elif not HAVE_INFLUX:
                why = "package not installed"
            else:
                why = "missing token"
            self.logger.info("InfluxDB writes are OFF (%s).", why)

        # Optional environmental register configs (set from CLI)
        self.temp_cfg: Optional[Dict[str, Any]] = None
        self.rh_cfg: Optional[Dict[str, Any]] = None

    # ----- Modbus helpers (version-proof + retry for transient errors) -----
    def _read_h(self, start_reg: int, count: int = 1) -> Optional[List[int]]:
        addr = h_addr(start_reg)
        fn = self.client.read_holding_registers
        for attempt in range(3):
            try:
                rr = fn(address=addr, count=count, unit=self.unit_id)
            except TypeError:
                try:
                    rr = fn(address=addr, count=count, slave=self.unit_id)
                except TypeError:
                    rr = fn(address=addr, count=count)
            if rr.isError():
                code = getattr(rr, 'exception_code', None)
                # 6 = Device Busy; 11 = Gateway Target Device Failed to Respond
                if code in (6, 11) and attempt < 2:
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                self.logger.error(f"Read holding {start_reg} x{count} failed: {rr}")
                return None
            return list(rr.registers)
        return None

    def _write_h(self, reg: int, value: int) -> bool:
        addr = h_addr(reg)
        fn = self.client.write_register
        try:
            wr = fn(address=addr, value=value & 0xFFFF, unit=self.unit_id)
        except TypeError:
            try:
                wr = fn(address=addr, value=value & 0xFFFF, slave=self.unit_id)
            except TypeError:
                wr = fn(address=addr, value=value & 0xFFFF)
        if wr.isError():
            self.logger.error(f"Write holding {reg}={value} failed: {wr}")
            return False
        return True

    def _read_i(self, start_reg: int, count: int = 1) -> Optional[List[int]]:
        addr = i_addr(start_reg)
        fn = self.client.read_input_registers
        for attempt in range(3):
            try:
                rr = fn(address=addr, count=count, unit=self.unit_id)
            except TypeError:
                try:
                    rr = fn(address=addr, count=count, slave=self.unit_id)
                except TypeError:
                    rr = fn(address=addr, count=count)
            if rr.isError():
                code = getattr(rr, 'exception_code', None)
                if code in (6, 11) and attempt < 2:
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                self.logger.error(f"Read input {start_reg} x{count} failed: {rr}")
                return None
            return list(rr.registers)
        return None

    # Generic scalar input reader (for Temp/RH etc.)
    def _read_scalar_input(self, reg: int, dtype: str) -> Optional[float]:
        dtype = (dtype or "").lower()
        if dtype in ("u16", "i16"):
            regs = self._read_i(reg, 1)
            if not regs: return None
            v = regs[0] & 0xFFFF
            if dtype == "i16" and v >= 0x8000:
                v -= 0x10000
            return float(v)
        elif dtype in ("u32", "i32", "float", "float_be", "float_le"):
            regs = self._read_i(reg, 2)
            if not regs or len(regs) < 2: return None
            hi, lo = regs[0], regs[1]
            if dtype == "u32":
                return float(regs_to_u32(hi, lo))
            if dtype == "i32":
                return float(regs_to_i32(hi, lo))
            # float types
            be = True if dtype in ("float", "float_be") else False
            return float(regs_to_float_ieee(hi, lo, word_order_big_endian=be))
        else:
            self.logger.error(f"Unsupported dtype '{dtype}' for input read")
            return None

    # ----- Detection -----
    def detect_device(self) -> DeviceProfile:
        pn = self._read_h(APEX_PROFILE.PRODUCT_NAME_START, APEX_PROFILE.PRODUCT_NAME_LEN) or []
        mn = self._read_h(APEX_PROFILE.MODEL_NAME_START, APEX_PROFILE.MODEL_NAME_LEN) or []
        product = decode_ascii_words(pn) if pn else ''
        model = decode_ascii_words(mn) if mn else ''
        self.product_name = product
        self.model_name = model

        text = f"{product} {model}".upper()
        if "APEX" in text:
            self.profile = APEX_PROFILE
        elif "REMOTE" in text or "3014P" in text or "5014P" in text or "2014P" in text:
            self.profile = REMOTE_PROFILE
        else:
            self.profile = REMOTE_PROFILE

        ver = self._read_h(40001, 1)
        if ver:
            self.register_map_version = ver[0]

        self.logger.info(f"Detected device: type={self.profile.type} product='{product}' model='{model}' regmap={self.register_map_version}")
        return self.profile

    # ----- Actions: set times & start/stop -----
    def set_times(self, sample: Optional[int] = None, hold: Optional[int] = None,
                  delay: Optional[int] = None, save: bool = True) -> None:
        updated = False
        if sample is not None:
            self._write_h(40034, sample & 0xFFFF)   # Sample Seconds (low)
            updated = True
            self.logger.info("Set Sample=%s s", sample)
        if hold is not None:
            self._write_h(40032, hold & 0xFFFF)     # Hold Seconds (low)
            updated = True
            self.logger.info("Set Hold=%s s", hold)
        if delay is not None:
            self._write_h(40030, delay & 0xFFFF)    # Initial Delay Seconds (low)
            updated = True
            self.logger.info("Set Delay=%s s", delay)
        if updated and save:
            self._write_h(40002, 4)                 # Save instrument parameters
            self.logger.info("Saved parameters to device.")

    def start(self, wait: bool = True, timeout: float = 30.0) -> bool:
        self._write_h(40002, 11)  # Instrument Start
        self.logger.info("Sent START command (40002=11).")
        if not wait:
            return True
        t0 = time.time()
        while time.time() - t0 < timeout:
            ds = self._read_h(40003, 1)
            if ds:
                running = bool(ds[0] & (1 << 0))
                sampling = bool(ds[0] & (1 << 1))
                if running or sampling:
                    return True
            time.sleep(0.5)
        return False

    def stop(self) -> None:
        self._write_h(40002, 12)  # Instrument Stop
        self.logger.info("Sent STOP command (40002=12).")

    # ----- Polling -----
    def poll_once(self) -> Optional[Dict]:
        if not self.profile:
            self.detect_device()

        # Flow (CFM, /100 per docs)
        flow_regs = self._read_h(self.profile.FLOW_REG, 1)
        flow_cfm = None
        if flow_regs:
            flow_cfm = flow_regs[0] / 100.0

        # Device status
        ds_regs = self._read_h(self.profile.DEVICE_STATUS, 1)
        device_status_val = ds_regs[0] if ds_regs else 0
        device_status_bits = self._decode_device_status_bits(device_status_val, self.profile.type)

        # Apex extended status (40056)
        ext_status_val = None
        if self.profile.EXT_STATUS:
            ext_regs = self._read_h(self.profile.EXT_STATUS, 1)
            if ext_regs:
                ext_status_val = ext_regs[0]

        # Alarm flags
        alarms: Dict[str, int] = {}
        if self.profile.type == 'apex':
            for key, reg in (
                ("particle_alarm_high_flags", self.profile.ALARM_PARTICLE_HI),
                ("particle_alarm_low_flags", self.profile.ALARM_PARTICLE_LO),
                ("analog_alarm_high_flags", self.profile.ALARM_ANALOG_HI),
                ("analog_alarm_low_flags", self.profile.ALARM_ANALOG_LO),
            ):
                if reg:
                    regs = self._read_h(reg, 1)
                    if regs:
                        alarms[key] = regs[0]

        # Latest record: set 40025 = -1 (0xFFFF), then read 30001.. up to channels
        self._write_h(40025, 0xFFFF)  # -1 => last record
        data_regs = self._read_i(30001, 20)
        timestamp = None
        sample_time_s = None
        counts: Dict[str, int] = {}

        if data_regs and len(data_regs) >= 20:
            ts_high, ts_low = data_regs[0], data_regs[1]
            timestamp = regs_to_u32(ts_high, ts_low)
            st_high, st_low = data_regs[2], data_regs[3]
            sample_time_s = regs_to_u32(st_high, st_low)

            # Channels start at 30009 (index 8 in our slice). Up to max_channels.
            start_idx = 8
            for ch in range(1, self.max_channels + 1):
                idx = start_idx + (ch - 1) * 2
                if idx + 1 < len(data_regs):
                    high, low = data_regs[idx], data_regs[idx + 1]
                    counts[f"ch{ch}"] = regs_to_u32(high, low)

            # Sample Status (30007/8) – threshold bits for Remote P
            sample_status_high, sample_status_low = data_regs[6], data_regs[7]
            sample_status_val = regs_to_u32(sample_status_high, sample_status_low)
            if self.profile.type == 'remote':
                alarms["sample_status"] = sample_status_val
                alarms["threshold_high"] = (sample_status_val >> 4) & 1
                alarms["threshold_low"] = (sample_status_val >> 5) & 1

        # Optional environmental reads
        temp_c = None
        rh_percent = None
        if self.temp_cfg and self.temp_cfg.get("reg"):
            v = self._read_scalar_input(self.temp_cfg["reg"], self.temp_cfg.get("dtype", "float"))
            if v is not None:
                temp_c = (v * float(self.temp_cfg.get("scale", 1.0))) + float(self.temp_cfg.get("offset", 0.0))
        if self.rh_cfg and self.rh_cfg.get("reg"):
            v = self._read_scalar_input(self.rh_cfg["reg"], self.rh_cfg.get("dtype", "float"))
            if v is not None:
                rh_percent = (v * float(self.rh_cfg.get("scale", 1.0))) + float(self.rh_cfg.get("offset", 0.0))

        # Concentrations: particles per 0.1 ft^3
        conc_0p1ft3: Dict[str, float] = {}
        sample_volume_cuft = None
        try:
            if flow_cfm is not None and sample_time_s and sample_time_s > 0:
                sample_volume_cuft = (flow_cfm * float(sample_time_s)) / 60.0
                if sample_volume_cuft > 0:
                    for ch, cnt in counts.items():
                        conc_0p1ft3[ch] = float(cnt) * 0.1 / sample_volume_cuft
        except Exception as _:
            pass

        record = {
            "device_type": self.profile.type,
            "product_name": self.product_name,
            "model_name": self.model_name,
            "regmap_version": self.register_map_version,
            "flow_cfm": flow_cfm,
            "device_status_val": device_status_val,
            "device_status": device_status_bits,
            "ext_status_val": ext_status_val,
            "alarms": alarms,
            "timestamp": timestamp,
            "sample_time_s": sample_time_s,
            "counts": counts,
            "sample_volume_cuft": sample_volume_cuft,
            "conc_0p1ft3": conc_0p1ft3,   # particles per 0.1 ft^3
            "temp_c": temp_c,
            "rh_percent": rh_percent,
        }
        return record

    # ----- Decoders -----
    @staticmethod
    def _decode_device_status_bits(val: int, device_type: str) -> Dict[str, int]:
        bits = {
            "running": (val >> 0) & 1,
            "sampling": (val >> 1) & 1,
            "new_data": (val >> 2) & 1,
            "device_error": (val >> 3) & 1,
        }
        if device_type == 'apex':
            extra = {
                "data_validation": (val >> 9) & 1,
                "location_validation": (val >> 10) & 1,
                "laser_status": (val >> 11) & 1,
                "flow_status": (val >> 12) & 1,
                "service_status": (val >> 13) & 1,
                "threshold_high": (val >> 14) & 1,
                "threshold_low": (val >> 15) & 1,
            }
            bits.update(extra)
        return bits

    # ----- Influx write -----
    def write_influx(self, record: Dict):
        if not self.enable_influx or not self.write_api or not HAVE_INFLUX:
            return
        tags = {
            "device_type": record.get("device_type", ""),
            "product_name": record.get("product_name", ""),
            "model_name": record.get("model_name", ""),
            "host": self.host,
            "unit_id": str(self.unit_id),
            "count_uom": "particles/0.1 ft3",   # tag for concentration fields
        }
        fields = {}

        if record.get("flow_cfm") is not None:
            fields["flow_cfm"] = float(record["flow_cfm"])

        fields["device_status_val"] = int(record.get("device_status_val") or 0)
        for k, v in (record.get("device_status") or {}).items():
            fields[f"status_{k}"] = int(v)

        if record.get("ext_status_val") is not None:
            fields["ext_status_val"] = int(record["ext_status_val"])

        for k, v in (record.get("alarms") or {}).items():
            fields[f"alarm_{k}"] = int(v)

        # Raw counts
        for k, v in (record.get("counts") or {}).items():
            fields[f"count_{k}"] = int(v)

        # Concentrations per 0.1 ft^3
        for k, v in (record.get("conc_0p1ft3") or {}).items():
            fields[f"conc0p1ft3_{k}"] = float(v)

        # Sample volume (debug)
        if record.get("sample_volume_cuft") is not None:
            fields["sample_volume_cuft"] = float(record["sample_volume_cuft"])

        # Environmental
        if record.get("temp_c") is not None:
            fields["temp_c"] = float(record["temp_c"])
        if record.get("rh_percent") is not None:
            fields["rh_percent"] = float(record["rh_percent"])

        ts = record.get("timestamp")
        point = Point("lighthouse") \
            .tag("device_type", tags["device_type"]) \
            .tag("product_name", tags["product_name"]) \
            .tag("model_name", tags["model_name"]) \
            .tag("host", tags["host"]) \
            .tag("unit_id", tags["unit_id"]) \
            .tag("count_uom", tags["count_uom"])

        for fk, fv in fields.items():
            point = point.field(fk, fv)

        if ts is not None and ts > 0:
            point = point.time(int(ts), WritePrecision.S)
        else:
            point = point.time(time.time_ns(), WritePrecision.NS)

        self.write_api.write(bucket=self._influx_bucket, org=self._influx_org, record=point)

    # ----- Main loop -----
    def run(self):
        if not self.client.connect():
            self.logger.error(f"Unable to connect to {self.host}:{self.port}")
            sys.exit(2)

        try:
            self.detect_device()
            # Optionally set times and start sampling once on connect
            if getattr(self, 'auto_start', False):
                self.set_times(
                    sample=getattr(self, 'start_sample', 1),
                    hold=getattr(self, 'start_hold', 14),
                    delay=getattr(self, 'start_delay', 0),
                    save=getattr(self, 'save_params', True),
                )
                ok = self.start(wait=True, timeout=30.0)
                self.logger.info("Auto-start %s (running=%s)", "OK" if ok else "TIMED OUT", ok)
                time.sleep(1.0)  # brief settle

            while True:
                try:
                    rec = self.poll_once()
                    if rec:
                        self.write_influx(rec)
                        self.logger.debug(f"Wrote record: {rec}")
                except ModbusException as me:
                    self.logger.error(f"Modbus exception: {me}")
                except Exception as e:
                    self.logger.exception(f"Error during poll: {e}")
                time.sleep(self.interval)
        finally:
            self.client.close()
            try:
                if self.influx:
                    self.influx.__del__()  # close client
            except Exception:
                pass

# ----- CLI -----
def parse_args():
    p = argparse.ArgumentParser(description="Lighthouse ApexRp/Remote P Modbus TCP -> InfluxDB logger")
    p.add_argument("--host", required=True, help="Modbus TCP host/IP of the unit")
    p.add_argument("--port", type=int, default=502, help="Modbus TCP port (default 502)")
    p.add_argument("--unit-id", type=int, default=1, help="Modbus unit ID (default 1)")
    p.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds (default 5)")
    p.add_argument("--max-channels", type=int, default=6, help="Max particle channels to read (default 6)")
    p.add_argument("--verbose", action="store_true", help="Enable verbose (DEBUG) logging")
    p.add_argument("--log-file", default=None, help="Path to log file (optional)")

    # Auto-start and timing
    p.add_argument("--auto-start", action="store_true",
                   help="On startup, set times (Sample/Hold/Delay) and send START; wait for RUNNING.")
    p.add_argument("--sample", type=int, help="Sample seconds (used with --auto-start; default 1)")
    p.add_argument("--hold", type=int, help="Hold seconds (used with --auto-start; default 14)")
    p.add_argument("--delay", type=int, help="Initial delay seconds (used with --auto-start; default 0)")
    p.add_argument("--no-save-params", action="store_true",
                   help="Do not persist new timing values to device NVRAM.")
    # Convenience: minutes-based flags
    p.add_argument("--sample-min", type=float, help="Sample MINUTES (converted to seconds; used with --auto-start)")
    p.add_argument("--hold-min", type=float, help="Hold MINUTES (converted to seconds; used with --auto-start)")
    p.add_argument("--delay-min", type=float, help="Initial delay MINUTES (converted to seconds; used with --auto-start)")

    # Environmental inputs (input registers). Dtypes: u16, i16, u32, i32, float, float_be, float_le
    p.add_argument("--temp-reg", type=int, help="Input register for Temperature (e.g., 300xx base)")
    p.add_argument("--temp-type", default="float", help="Temp dtype: u16|i16|u32|i32|float|float_be|float_le (default float)")
    p.add_argument("--temp-scale", type=float, default=1.0, help="Temp scale multiplier (default 1.0)")
    p.add_argument("--temp-offset", type=float, default=0.0, help="Temp offset added after scaling (default 0.0)")
    p.add_argument("--rh-reg", type=int, help="Input register for RH% (e.g., 300xx base)")
    p.add_argument("--rh-type", default="float", help="RH dtype: u16|i16|u32|i32|float|float_be|float_le (default float)")
    p.add_argument("--rh-scale", type=float, default=1.0, help="RH scale multiplier (default 1.0)")
    p.add_argument("--rh-offset", type=float, default=0.0, help="RH offset added after scaling (default 0.0)")

    # Influx v2
    p.add_argument("--influx-url", default=os.environ.get("INFLUX_URL", "http://localhost:8086"))
    p.add_argument("--influx-org", default=os.environ.get("INFLUX_ORG", "default-org"))
    p.add_argument("--influx-bucket", default=os.environ.get("INFLUX_BUCKET", "lighthouse"))
    p.add_argument("--influx-token", default=os.environ.get("INFLUX_TOKEN", ""))
    p.add_argument("--no-influx", action="store_true", help="Disable InfluxDB writes (run without Influx)")
    args = p.parse_args()

    if not args.influx_token and not args.no_influx:
        print("[INFO] No InfluxDB token provided. Influx writes will be disabled unless you pass a token or use --no-influx.", file=sys.stderr)
    return args

def main():
    args = parse_args()
    logger = LighthouseLogger(
        host=args.host,
        port=args.port,
        unit_id=args.unit_id,
        influx_url=args.influx_url,
        influx_org=args.influx_org,
        influx_bucket=args.influx_bucket,
        influx_token=args.influx_token,
        interval=args.interval,
        max_channels=args.max_channels,
        verbose=args.verbose,
        log_file=args.log_file,
        enable_influx=not args.no_influx,
    )

    # Configure optional auto-start behavior
    logger.auto_start = bool(args.auto_start)
    def _mins_to_secs(x): return None if x is None else int(round(float(x) * 60.0))
    logger.start_sample = (1 if args.sample is None and args.sample_min is None else
                           (args.sample if args.sample is not None else _mins_to_secs(args.sample_min)))
    logger.start_hold   = (14 if args.hold is None and args.hold_min is None else
                           (args.hold if args.hold is not None else _mins_to_secs(args.hold_min)))
    logger.start_delay  = (0 if args.delay is None and args.delay_min is None else
                           (args.delay if args.delay is not None else _mins_to_secs(args.delay_min)))
    logger.save_params  = not args.no_save_params

    # Optional environmental registers
    if args.temp_reg:
        logger.temp_cfg = {"reg": int(args.temp_reg), "dtype": args.temp_type,
                           "scale": args.temp_scale, "offset": args.temp_offset}
    if args.rh_reg:
        logger.rh_cfg = {"reg": int(args.rh_reg), "dtype": args.rh_type,
                         "scale": args.rh_scale, "offset": args.rh_offset}

    logger.run()

if __name__ == "__main__":
    main()
