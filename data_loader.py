"""
data_loader.py
Updated to fix Pandas Index clipping bugs and perfectly parse 
the Kaggle 'cleaned_dataset' metadata structure or the original .mat files.
"""

from imports import *
from config import DATA_DIR, EOL_THRESHOLD

def download_dataset():
    """Download NASA Battery Dataset from Kaggle."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.system(f'python -m kaggle datasets download -d patrickfleith/nasa-battery-dataset -p {DATA_DIR} --unzip')
    all_files = glob.glob(os.path.join(DATA_DIR, '**', '*'), recursive=True)
    print(f"\nTotal files downloaded: {len(all_files)}")

def load_kaggle_cleaned_csvs(meta_path):
    """
    Parses the Kaggle cleaned_dataset structure where metadata.csv acts as a manifest 
    for thousands of individual {uid}.csv files in the data/ folder.
    """
    data_folder = os.path.join(os.path.dirname(meta_path), 'data')
    
    # Read metadata without forcing dtype yet to avoid hidden space issues
    df_meta = pd.read_csv(meta_path)
    df_meta.columns = df_meta.columns.str.strip() 
    
    # Bulletproof UID matching
    uid_col = next((c for c in df_meta.columns if c.lower() in ['uid', 'id', 'filename']), None)
    if uid_col:
        df_meta[uid_col] = df_meta[uid_col].astype(str).str.replace(r'\.0$', '', regex=True)
        df_meta[uid_col] = df_meta[uid_col].apply(lambda x: x.zfill(5) if x.isdigit() else x)
    
    # Filter for discharge cycles safely
    type_col = next((c for c in df_meta.columns if 'type' in c.lower()), None)
    if type_col:
        df_meta[type_col] = df_meta[type_col].astype(str).str.lower().str.strip()
        df_meta = df_meta[df_meta[type_col] == 'discharge'].copy()
    
    if 'start_time' in df_meta.columns:
        df_meta['start_time'] = pd.to_datetime(df_meta['start_time'], errors='coerce')
        df_meta = df_meta.sort_values(['battery_id', 'start_time'])
        
    cap_col_meta = next((c for c in df_meta.columns if 'capacity' in c.lower()), None)
    
    if cap_col_meta:
        df_meta[cap_col_meta] = df_meta[cap_col_meta].astype(str).str.strip('[]')
        df_meta[cap_col_meta] = pd.to_numeric(df_meta[cap_col_meta], errors='coerce')
    
    all_records = []
    
    for bat_id, grp in df_meta.groupby('battery_id'):
        grp = grp.reset_index(drop=True)
        valid_caps = grp[cap_col_meta].dropna() if cap_col_meta else []
        nominal_cap = valid_caps.iloc[0] if len(valid_caps) > 0 else np.nan
        
        for cycle_idx, row in grp.iterrows():
            uid = row.get(uid_col)
            cap = row[cap_col_meta] if cap_col_meta else np.nan
            
            csv_file = os.path.join(data_folder, f"{uid}.csv")
            if not os.path.exists(csv_file):
                continue
                
            df_ts = pd.read_csv(csv_file)
            cols_lower = {c.lower(): c for c in df_ts.columns}
            
            v_col = next((cols_lower[c] for c in cols_lower if 'voltage' in c and 'load' not in c), None)
            i_col = next((cols_lower[c] for c in cols_lower if 'current' in c and 'load' not in c), None)
            t_col = next((cols_lower[c] for c in cols_lower if 'temp' in c), None)
            vl_col = next((cols_lower[c] for c in cols_lower if 'voltage' in c and 'load' in c), None)
            il_col = next((cols_lower[c] for c in cols_lower if 'current' in c and 'load' in c), None)
            time_col = next((cols_lower[c] for c in cols_lower if 'time' in c), None)

            def safe_stats(col_name):
                if col_name and col_name in df_ts.columns:
                    arr = np.array(df_ts[col_name].dropna())
                    if len(arr) > 0:
                        return [np.mean(arr), np.std(arr), np.min(arr), np.max(arr), np.median(arr), arr[-1]-arr[0]]
                return [0.0] * 6

            rec = {
                'battery_id': bat_id,
                'cycle_index': cycle_idx,
                'capacity': cap,
                'nominal_capacity': nominal_cap
            }

            for actual_col, prefix in [(v_col, 'voltage'), (i_col, 'current'), (t_col, 'temp'), (vl_col, 'vol_load'), (il_col, 'cur_load')]:
                stats = safe_stats(actual_col)
                for stat_name, val in zip(['mean', 'std', 'min', 'max', 'med', 'delta'], stats):
                    rec[f'{prefix}_{stat_name}'] = val

            if time_col and time_col in df_ts.columns:
                rec['discharge_time'] = float(df_ts[time_col].iloc[-1] - df_ts[time_col].iloc[0])
            else:
                rec['discharge_time'] = float(len(df_ts))

            if v_col and i_col:
                power = np.abs(df_ts[v_col] * df_ts[i_col])
                if time_col and time_col in df_ts.columns:
                    time_arr = df_ts[time_col].values
                    dt = np.diff(time_arr)
                    dt = np.append(dt, dt[-1]) if len(dt) > 0 else np.ones(len(power))
                    rec['energy_discharged'] = float(np.trapz(power, dx=np.mean(dt)))
                else:
                    rec['energy_discharged'] = float(np.trapz(power))
                rec['peak_power'] = float(power.max())
            else:
                rec['energy_discharged'] = 0.0
                rec['peak_power'] = 0.0

            all_records.append(rec)
            
    if len(all_records) == 0:
        sample_uid = df_meta[uid_col].iloc[0] if uid_col else "UNKNOWN"
        raise ValueError(f"Could not link metadata to files. Looked for {data_folder}\\{sample_uid}.csv but found nothing.")
        
    df_all = pd.DataFrame(all_records).dropna(subset=['capacity'])
    df_all['SOH'] = (df_all['capacity'] / df_all['nominal_capacity'] * 100).clip(0, 100)

    # ── THE FIX: Using np.clip to safely handle Pandas Index subtraction ───
    def calc_rul(grp):
        eol_indices = grp.index[grp['SOH'] < EOL_THRESHOLD].tolist()
        if eol_indices:
            eol_idx = eol_indices[0]
            grp['RUL'] = np.clip(eol_idx - grp.index, a_min=0, a_max=None)
        else:
            grp['RUL'] = np.clip(grp.index[-1] - grp.index, a_min=0, a_max=None)
        return grp
    # ───────────────────────────────────────────────────────────────────────

    df_all = df_all.groupby('battery_id', group_keys=False).apply(calc_rul)
    return df_all.reset_index(drop=True)

def parse_nasa_mat(filepath):
    """Parse original NASA .mat file structure."""
    mat = loadmat(filepath, simplify_cells=True)
    batt_name = [k for k in mat.keys() if not k.startswith('_')][0]
    battery = mat[batt_name]

    cycle_data = battery['cycle']
    records = []

    for i, cycle in enumerate(cycle_data):
        c_type = cycle.get('type', '')
        if c_type != 'discharge':
            continue

        data = cycle.get('data', {})
        if not data:
            continue

        voltage    = np.atleast_1d(data.get('Voltage_measured', []))
        current    = np.atleast_1d(data.get('Current_measured', []))
        temp       = np.atleast_1d(data.get('Temperature_measured', []))
        cur_load   = np.atleast_1d(data.get('Current_load', []))
        vol_load   = np.atleast_1d(data.get('Voltage_load', []))
        time_s     = np.atleast_1d(data.get('Time', []))
        capacity   = data.get('Capacity', np.nan)

        if len(voltage) < 5:
            continue

        def safe_stats(arr):
            if len(arr) == 0: return [np.nan] * 6
            return [np.mean(arr), np.std(arr), np.min(arr), np.max(arr), np.median(arr), arr[-1] - arr[0]]

        record = {
            'cycle_index': i,
            'capacity': float(capacity) if capacity else np.nan,
            'discharge_time': float(time_s[-1] - time_s[0]) if len(time_s) > 1 else np.nan,
        }

        for arr, prefix in [(voltage, 'voltage'), (current, 'current'), (temp, 'temp'), (cur_load, 'cur_load'), (vol_load, 'vol_load')]:
            stats = safe_stats(arr)
            for stat_name, val in zip(['mean', 'std', 'min', 'max', 'med', 'delta'], stats):
                record[f'{prefix}_{stat_name}'] = val

        if len(voltage) == len(current) and len(voltage) > 1:
            power = np.abs(voltage * current)
            dt = np.diff(time_s) if len(time_s) == len(voltage) else np.ones(len(voltage) - 1)
            record['energy_discharged'] = float(np.trapz(power, dx=np.mean(dt)))
            record['peak_power'] = float(np.max(power))
        else:
            record['energy_discharged'] = np.nan
            record['peak_power'] = np.nan

        records.append(record)

    df = pd.DataFrame(records).dropna(subset=['capacity']).reset_index(drop=True)
    if len(df) == 0:
        return df
        
    nominal_capacity = df['capacity'].iloc[0]
    df['SOH'] = (df['capacity'] / nominal_capacity * 100).clip(0, 100)
    
    # ── THE FIX applied here as well for backwards compatibility ───
    eol_indices = df.index[df['SOH'] < EOL_THRESHOLD].tolist()
    if eol_indices:
        df['RUL'] = np.clip(eol_indices[0] - df.index, a_min=0, a_max=None)
    else:
        df['RUL'] = np.clip(df.index[-1] - df.index, a_min=0, a_max=None)

    df['battery_id'] = os.path.basename(filepath).replace('.mat', '')
    df['nominal_capacity'] = nominal_capacity
    return df

def load_all_batteries():
    """Load data from Kaggle cleaned CSVs or fallback to .mat files."""
    
    meta_paths = glob.glob(os.path.join(DATA_DIR, '**', 'metadata.csv'), recursive=True)
    if meta_paths:
        print(f"Detected Kaggle 'cleaned_dataset' manifest: {os.path.basename(meta_paths[0])}")
        print("Processing thousands of individual cycle CSV files... (this will take a moment)")
        
        df_all = load_kaggle_cleaned_csvs(meta_paths[0])
        
        if df_all is not None and len(df_all) > 0:
            print(f"  ✅ Parsed {len(df_all)} total discharge cycles successfully.")
            print(f"\n📊 Combined dataset summary:")
            print(df_all[['battery_id', 'cycle_index', 'capacity', 'SOH', 'RUL']].describe())
            return df_all
            
    mat_files = glob.glob(os.path.join(DATA_DIR, '**', '*.mat'), recursive=True)
    if mat_files:
        print(f"Found {len(mat_files)} .mat files. Parsing...")
        all_dfs = []
        for f in sorted(mat_files):
            try:
                df_bat = parse_nasa_mat(f)
                if len(df_bat) > 10:
                    all_dfs.append(df_bat)
                    print(f"  ✅ {os.path.basename(f):20s} | {len(df_bat):3d} cycles | "
                          f"SOH range: {df_bat['SOH'].min():.1f}% – {df_bat['SOH'].max():.1f}%")
            except Exception as e:
                print(f"  ⚠️  {os.path.basename(f):20s} | Skipped: {e}")
                
        if all_dfs:
            df_all = pd.concat(all_dfs, ignore_index=True)
            print(f"\n📊 Combined dataset: {len(df_all)} discharge cycles from {df_all['battery_id'].nunique()} batteries")
            print(df_all[['battery_id', 'cycle_index', 'capacity', 'SOH', 'RUL']].describe())
            return df_all
            
    raise ValueError(f"No valid battery data (.mat or metadata.csv) found in {DATA_DIR}.")