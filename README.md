[README.md](https://github.com/user-attachments/files/21879884/README.md)
# Lighthouse Modbus TCP Logger (CLI)

Poll Lighthouse ApexRp / Remote P counters over **Modbus TCP** and optionally write to **InfluxDB v2**.
Computes per-channel **concentration in _particles per 0.1 ft³_** from counts + flow + sample time.
Also supports optional **Temperature (°C)** and **RH (%)** via input registers.

## Features
- Auto-detect **ApexRp** vs **REMOTE 2014P/3014P/5014P**
- Strictly **Modbus TCP** (no serial)
- **Auto-start**: set Sample/Hold/Delay and send `START`
- Seconds **or minutes** flags for timing
- Optional **InfluxDB v2** (toggle with `--no-influx`)
- Optional **Temp/RH** reads with scaling/offset
- Resilient Modbus usage: works with `unit=`/`slave=` pymodbus variants + quick retries

## Install
```bash
python -m venv .venv && source .venv/bin/activate
# Full (Modbus + InfluxDB v2 client)
pip install -r requirements.txt
# or minimal (no InfluxDB client)
# pip install -r requirements-min.txt
```

## Run (no Influx, auto-start with minutes)
```bash
python lighthouse_modbus_logger.py   --host 192.168.50.125 --unit-id 1 --interval 5   --auto-start --sample-min 1 --hold-min 14 --delay-min 0   --no-influx --verbose --log-file ./lighthouse.log
```

## With InfluxDB v2
```bash
export INFLUX_TOKEN="YOUR_TOKEN"
python lighthouse_modbus_logger.py   --host 192.168.50.125 --unit-id 1 --interval 5   --auto-start --sample 60 --hold 0 --delay 0   --influx-url http://localhost:8086   --influx-org myorg   --influx-bucket cleanroom   --influx-token $INFLUX_TOKEN   --verbose --log-file ./lighthouse.log
```

## Environmental inputs (optional)
If your device exposes Temp/RH on input registers:
```bash
python lighthouse_modbus_logger.py   --host 192.168.50.125 --unit-id 1   --temp-reg 30071 --temp-type float --temp-scale 1.0 --temp-offset 0.0   --rh-reg 30073 --rh-type float --rh-scale 1.0 --rh-offset 0.0   --no-influx
```
> Supported dtypes: `u16, i16, u32, i32, float, float_be, float_le`

## Output (Influx fields/tags)
- Tags: `device_type`, `product_name`, `model_name`, `host`, `unit_id`, `count_uom="particles/0.1 ft3"`
- Fields: 
  - `flow_cfm`, `device_status_val`, `status_*` bits, `ext_status_val` (Apex)
  - `alarm_*`, `count_ch*` (raw counts)
  - `conc0p1ft3_ch*` (**particles/0.1 ft³**)
  - `sample_volume_cuft` (derived), `temp_c`, `rh_percent` (optional)

## Notes
- Timing registers are **seconds** on the device; use `--*-min` for convenience.
- Sample volume (ft³) = `flow_cfm * sample_time_s / 60`. 
- `conc0p1ft3_chN = counts_chN * 0.1 / sample_volume_cuft`.

## Quick Git push
```bash
unzip lighthouse-cli-logger.zip && cd lighthouse-cli-logger
git init && git add . && git commit -m "Initial commit: CLI logger"
# create a repo on GitHub, then:
git branch -M main
git remote add origin https://github.com/<owner>/<repo>.git
git push -u origin main
```
