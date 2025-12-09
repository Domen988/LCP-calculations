import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from pvlib import location
from typing import Tuple

# --- CONFIGURATION CLASS ---
@dataclass
class PanelGeometry:
    width: float = 1.0
    length: float = 1.0
    thickness: float = 0.1
    pitch_x: float = 1.05
    pitch_y: float = 1.05
    # Vector from Pivot Point TO Geometric Center
    pivot_offset: Tuple[float, float, float] = (0.0, 0.0, 0.2)

class ClashSimulator:
    def __init__(self, geo: PanelGeometry, lat: float, lon: float, frame_rot: float):
        self.geo = geo
        self.location = location.Location(lat, lon, tz='Africa/Johannesburg')
        self.frame_rot = frame_rot

    def _get_rotation_matrices(self, az_series: np.ndarray, el_series: np.ndarray) -> np.ndarray:
        # Adjust Azimuth to Frame Orientation
        az_frame = az_series - self.frame_rot
        theta = np.radians(90 - az_frame)
        c, s = np.cos(theta), np.sin(theta)
        zeros, ones = np.zeros_like(c), np.ones_like(c)

        # Rz (Azimuth) and Rx (Tilt)
        Rz = np.array([[c, -s, zeros], [s, c, zeros], [zeros, zeros, ones]]).transpose(2, 0, 1)
        
        tilt = np.radians(90 - el_series)
        ct, st = np.cos(tilt), np.sin(tilt)
        Rx = np.array([[ones, zeros, zeros], [zeros, ct, -st], [zeros, st, ct]]).transpose(2, 0, 1)

        return np.matmul(Rz, Rx)

    def run_year(self, year: int = 2024, frequency: str = '6min'):
        print(f"1. Generating Solar Data ({frequency})...")
        times = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:59', freq=frequency, tz=self.location.tz)
        solpos = self.location.get_solarposition(times)
        
        # Filter Daylight
        solpos = solpos[solpos['apparent_elevation'] > 0].copy()
        
        print("2. Calculating Clashes...")
        az = solpos['azimuth'].values
        el = solpos['apparent_elevation'].values
        R = self._get_rotation_matrices(az, el)
        R_inv = R.transpose(0, 2, 1)

        # Geometry Setup
        neighbors = np.array([
            [self.geo.pitch_x, 0, 0], [-self.geo.pitch_x, 0, 0],
            [0, self.geo.pitch_y, 0], [0, -self.geo.pitch_y, 0]
        ])
        
        ox, oy, oz = self.geo.pivot_offset
        w, l, t = self.geo.width, self.geo.length, self.geo.thickness
        b_min = np.array([ox - w/2, oy - l/2, oz - t/2])
        b_max = np.array([ox + w/2, oy + l/2, oz + t/2])

        # Vectorized Clash Check
        clashes = np.zeros(len(az), dtype=bool)
        for n_vec in neighbors:
            n_batch = np.tile(n_vec, (len(az), 1))
            d_local = np.einsum('nij,nj->ni', R_inv, n_batch)
            
            ov_x = np.maximum(b_min[0], b_min[0]+d_local[:,0]) < np.minimum(b_max[0], b_max[0]+d_local[:,0])
            ov_y = np.maximum(b_min[1], b_min[1]+d_local[:,1]) < np.minimum(b_max[1], b_max[1]+d_local[:,1])
            ov_z = np.maximum(b_min[2], b_min[2]+d_local[:,2]) < np.minimum(b_max[2], b_max[2]+d_local[:,2])
            
            clashes |= (ov_x & ov_y & ov_z)

        return solpos, clashes

    def save_results(self, df: pd.DataFrame, clashes: np.ndarray, filename_base: str):
        # 1. Save Data Table (CSV)
        output = df[['azimuth', 'apparent_elevation']].copy()
        output['Clash'] = clashes
        output.columns = ['Azimuth', 'Elevation', 'Clash']
        output.index.name = 'Time'
        output.reset_index(inplace=True)
        output.to_csv(f"{filename_base}.csv", index=False)
        
        # 2. Save Configuration (JSON)
        config = {
            "geometry": asdict(self.geo),
            "location": {
                "frame_rotation": self.frame_rot,
                "lat": self.location.latitude,
                "lon": self.location.longitude
            }
        }
        with open(f"{filename_base}.json", "w") as f:
            json.dump(config, f, indent=4)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Example: Pivot is 5cm below the center of the panel
    geo = PanelGeometry(pivot_offset=(0.0, 0.0, 0.05)) 
    sim = ClashSimulator(geo, lat=-25.86, lon=26.90, frame_rot=5.0)
    
    # --- RUN ---
    df_res, clash_res = sim.run_year()
    sim.save_results(df_res, clash_res, "simulation_results")
    print("Done. Saved .csv and .json files.")