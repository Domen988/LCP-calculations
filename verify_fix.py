import simulation_engine as sim_engine
import pandas as pd
import numpy as np

def verify():
    print("Verifying Simulation Engine Fix...")
    try:
        # Create minimal config
        geo = sim_engine.PanelGeometry()
        sim = sim_engine.ClashSimulator(geo, lat=-25.86, lon=26.90, frame_rot=5.0)
        
        # Run Short Simulation (checking run_year_power structure)
        # Using a dummy excel path might fail if file logic strict.
        # But run_year_power has `excel_path=None` default? 
        # Wait, my code does `if excel: load; else: no load (use clear sky fallback if implemented?)`
        # My `get_dni` implementation handles `None` DNI interpolator by falling back to clearsky.
        
        print("Running run_year_power (no excel)...")
        df, _ = sim.run_year_power(year=2024, frequency='1D', excel_path=None)
        
        print("Columns:", df.columns)
        if 'Time' in df.columns:
            print("SUCCESS: 'Time' column found.")
        else:
            print("FAILURE: 'Time' column missing.")

        if 'Azimuth' in df.columns and 'Elevation' in df.columns:
            print("SUCCESS: 'Azimuth' and 'Elevation' columns found (Capitalized).")
        else:
            print("FAILURE: Columns missing or lowercase.")
            
        if 'Power_Total_kW' in df.columns:
            print("SUCCESS: 'Power_Total_kW' column found.")
        else:
            print("FAILURE: 'Power_Total_kW' column missing.")
            
        if 'Clash' in df.columns:
            print("SUCCESS: 'Clash' column found.")
        else:
            print("FAILURE: 'Clash' column missing.")
            
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
