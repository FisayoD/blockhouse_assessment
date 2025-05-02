import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path, time_interval='1min'):
    df = pd.read_csv(file_path)
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values(['instrument_id', 'ts_event'])
    df.set_index('ts_event', inplace=True)
    return df

def calculate_order_flows(df, level=1):
    df_out = df.copy()
    bid_price_col = f'bid_px_{level-1:02d}'
    ask_price_col = f'ask_px_{level-1:02d}'
    bid_size_col = f'bid_sz_{level-1:02d}'
    ask_size_col = f'ask_sz_{level-1:02d}'
    
    for instrument in df_out['instrument_id'].unique():
        inst_mask = (df_out['instrument_id'] == instrument)
        inst_df = df_out[inst_mask].copy()
        
        inst_df[f'prev_{bid_price_col}'] = inst_df[bid_price_col].shift(1)
        inst_df[f'prev_{ask_price_col}'] = inst_df[ask_price_col].shift(1)
        inst_df[f'prev_{bid_size_col}'] = inst_df[bid_size_col].shift(1)
        inst_df[f'prev_{ask_size_col}'] = inst_df[ask_size_col].shift(1)
        
        conditions_bid = [
            inst_df[bid_price_col] > inst_df[f'prev_{bid_price_col}'],
            inst_df[bid_price_col] == inst_df[f'prev_{bid_price_col}'],
            inst_df[bid_price_col] < inst_df[f'prev_{bid_price_col}']
        ]
        
        choices_bid = [
            inst_df[bid_size_col],
            inst_df[bid_size_col] - inst_df[f'prev_{bid_size_col}'],
            -inst_df[f'prev_{bid_size_col}']
        ]
        
        inst_df[f'OF_bid_{level}'] = np.select(conditions_bid, choices_bid, default=0)
        
        conditions_ask = [
            inst_df[ask_price_col] > inst_df[f'prev_{ask_price_col}'],
            inst_df[ask_price_col] == inst_df[f'prev_{ask_price_col}'],
            inst_df[ask_price_col] < inst_df[f'prev_{ask_price_col}']
        ]
        
        choices_ask = [
            -inst_df[ask_size_col],
            inst_df[ask_size_col] - inst_df[f'prev_{ask_size_col}'],
            inst_df[ask_size_col]
        ]
        
        inst_df[f'OF_ask_{level}'] = np.select(conditions_ask, choices_ask, default=0)
        
        df_out.loc[inst_mask, f'OF_bid_{level}'] = inst_df[f'OF_bid_{level}']
        df_out.loc[inst_mask, f'OF_ask_{level}'] = inst_df[f'OF_ask_{level}']
    
    cols_to_drop = [col for col in df_out.columns if col.startswith('prev_')]
    df_out = df_out.drop(columns=cols_to_drop)
    
    return df_out

def calculate_best_level_OFI(df, time_interval='1min'):
    df_with_flows = calculate_order_flows(df, level=1)
    ofi_df = pd.DataFrame()
    
    for instrument in df_with_flows['instrument_id'].unique():
        inst_mask = (df_with_flows['instrument_id'] == instrument)
        inst_df = df_with_flows[inst_mask].copy()
        
        inst_resampled = inst_df.resample(time_interval).apply({
            'OF_bid_1': 'sum',
            'OF_ask_1': 'sum',
            'instrument_id': 'first',
            'symbol': 'first'
        })
        
        inst_resampled['OFI_1'] = inst_resampled['OF_bid_1'] - inst_resampled['OF_ask_1']
        ofi_df = pd.concat([ofi_df, inst_resampled])
    
    return ofi_df

def calculate_multi_level_OFI(df, max_level=10, time_interval='1min'):
    ofi_df = pd.DataFrame()
    
    for instrument in df['instrument_id'].unique():
        inst_mask = (df['instrument_id'] == instrument)
        inst_df = df[inst_mask].copy()
        
        for level in range(1, max_level + 1):
            inst_df = calculate_order_flows(inst_df, level=level)
        
        multi_level_ofi = {}
        for level in range(1, max_level + 1):
            resampled = inst_df.resample(time_interval).apply({
                f'OF_bid_{level}': 'sum',
                f'OF_ask_{level}': 'sum',
                'instrument_id': 'first',
                'symbol': 'first'
            })
            
            resampled[f'OFI_{level}'] = resampled[f'OF_bid_{level}'] - resampled[f'OF_ask_{level}']
            
            depth_cols = [
                f'bid_sz_{i:02d}' for i in range(max_level)
            ] + [
                f'ask_sz_{i:02d}' for i in range(max_level)
            ]
            
            avg_depth = inst_df[depth_cols].mean(axis=1)
            resampled_depth = avg_depth.resample(time_interval).mean()
            
            resampled[f'ofi_{level}'] = resampled[f'OFI_{level}'] / resampled_depth
            
            multi_level_ofi[level] = resampled[f'ofi_{level}']
        
        inst_multi_ofi = pd.concat(multi_level_ofi.values(), axis=1)
        inst_multi_ofi.columns = [f'ofi_{i}' for i in range(1, max_level + 1)]
        
        inst_multi_ofi['instrument_id'] = instrument
        inst_multi_ofi['symbol'] = inst_df['symbol'].iloc[0]
        
        ofi_df = pd.concat([ofi_df, inst_multi_ofi])
    
    return ofi_df

def calculate_integrated_OFI(multi_level_ofi, lookback_window=30):
    integrated_ofi = pd.DataFrame(index=multi_level_ofi.index)
    
    for instrument in multi_level_ofi['instrument_id'].unique():
        inst_mask = (multi_level_ofi['instrument_id'] == instrument)
        inst_df = multi_level_ofi[inst_mask].copy()
        
        feature_cols = [col for col in inst_df.columns if col.startswith('ofi_')]
        
        for i in range(lookback_window, len(inst_df)):
            window_data = inst_df[feature_cols].iloc[i-lookback_window:i]
            
            pca = PCA(n_components=1)
            pca.fit(window_data)
            
            w1 = pca.components_[0]
            
            w1_normalized = w1 / np.sum(np.abs(w1))
            
            current_ofi = inst_df[feature_cols].iloc[i]
            integrated_value = np.dot(current_ofi, w1_normalized)
            
            integrated_ofi.loc[inst_df.index[i], 'integrated_ofi'] = integrated_value
        
        integrated_ofi.loc[inst_df.index, 'instrument_id'] = instrument
        integrated_ofi.loc[inst_df.index, 'symbol'] = inst_df['symbol'].iloc[0]
    
    return integrated_ofi

def calculate_cross_asset_OFI(best_level_ofi, integrated_ofi):
    best_level_pivot = best_level_ofi.pivot(columns='symbol', values='OFI_1')
    best_level_pivot.columns = [f'best_ofi_{col}' for col in best_level_pivot.columns]
    
    integrated_pivot = integrated_ofi.pivot(columns='symbol', values='integrated_ofi')
    integrated_pivot.columns = [f'integrated_ofi_{col}' for col in integrated_pivot.columns]
    
    cross_asset_OFI = pd.concat([best_level_pivot, integrated_pivot], axis=1)
    
    return cross_asset_OFI

def main():
    file_path = 'first_25000_rows.csv'
    df = load_and_preprocess_data(file_path, time_interval='1min')
    
    best_level_ofi = calculate_best_level_OFI(df, time_interval='1min')
    print("Completed Best-level OFI calculations.")
    
    multi_level_ofi = calculate_multi_level_OFI(df, max_level=10, time_interval='1min')
    print("Completed multi_level_ofi calculations.")
    
    integrated_ofi = calculate_integrated_OFI(multi_level_ofi, lookback_window=30)
    print("Completed integrated_ofi calculations.")
    
    cross_asset_ofi = calculate_cross_asset_OFI(best_level_ofi, integrated_ofi)
    print("Completed cross_asset_ofi data preparation.")
    
    best_level_ofi.to_csv('best_level_ofi.csv')
    multi_level_ofi.to_csv('multi_level_ofi.csv')
    integrated_ofi.to_csv('integrated_ofi.csv')
    cross_asset_ofi.to_csv('cross_asset_ofi.csv')
    
    print("Done...Results are all saved.")

if __name__ == "__main__":
    main()