import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from pvlib import location
from typing import Tuple, List, Dict
import openpyxl
from scipy.interpolate import RegularGridInterpolator

# --- CONFIGURATION CLASS ---
@dataclass
class PanelGeometry:
    width: float = 1.0
    length: float = 1.0
    thickness: float = 0.1
    pitch_x: float = 1.05
    pitch_y: float = 1.05
    # Dimensions of the entire Plant Layout
    frame_spacing_x: float = 6.0 # Center-to-Center spacing of Frames E-W
    frame_spacing_y: float = 6.0 # Center-to-Center spacing of Frames N-S
    # Vector from Pivot Point TO Geometric Center
    pivot_offset: Tuple[float, float, float] = (0.0, 0.0, 0.2)

class ClashSimulator:
    def __init__(self, geo: PanelGeometry, lat: float, lon: float, frame_rot: float, frame_tilts: List[float] = [0.0, 0.0, 0.0]):
        self.geo = geo
        self.location = location.Location(lat, lon, tz='Africa/Johannesburg')
        self.frame_rot = frame_rot
        self.frame_tilts = frame_tilts # Tilt for Row 1, Row 2, Row 3 (Degrees)
        self.dni_interpolator = None

    def load_dni_data(self, excel_path: str):
        """Loads DNI data from the specific Excel format provided."""
        try:
            # Load Hourly Profiles (Month x Hour)
            df = pd.read_excel(excel_path, sheet_name='Hourly_profiles', skiprows=3)
            # Structure expected: Column 1 is labels, Cols 2-13 are Jan-Dec. Rows 4-27 are hours 0-1 to 23-24.
            # We need to extract the 24x12 matrix.
            # Based on inspection:
            # Rows 4 to 27 (indices in 0-based likely 3 to 26 if skiprows handled correctly?)
            # Let's be robust.
            
            # Re-read with specific range if possible, or filter.
            # The 'Unnamed: 0' col has '0 - 1', '1 - 2'...
            valid_rows = df[df.iloc[:, 0].astype(str).str.contains(r'\d+ - \d+', na=False)]
            if valid_rows.empty:
                print("Warning: Could not parse DNI data rows. Using default 0.")
                self.dni_interpolator = lambda x: np.zeros(len(x))
                return

            data_matrix = valid_rows.iloc[:, 2:14].values.astype(float) # Jan-Dec
            # data_matrix is 24 rows (hours) x 12 columns (months)
            
            # Create Interpolator
            # x: Month (1-12), y: Hour (0.5 to 23.5)
            months = np.arange(1, 13)
            hours = np.arange(0.5, 24.5, 1.0)
            self.dni_interpolator = RegularGridInterpolator((hours, months), data_matrix, bounds_error=False, fill_value=0)
            print(f"DNI Data Loaded. Matrix Shape: {data_matrix.shape}")
        except Exception as e:
            print(f"Error loading DNI data: {e}")
            self.dni_interpolator = None

    def get_dni(self, times: pd.DatetimeIndex) -> np.ndarray:
        if self.dni_interpolator is None:
             # Fallback to Clear Sky if no file
             cs = self.location.get_clearsky(times)
             return cs['dni'].values
        
        # Interpolate
        # We need (Hour, Month) points
        # Time decimal hour
        h = times.hour + times.minute / 60.0
        m = times.month
        pts = np.column_stack((h, m))
        return self.dni_interpolator(pts)

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
        return solpos, clashes

    def _calculate_clashes(self, az: np.ndarray, el: np.ndarray) -> np.ndarray:
        """Core Clash Logic for Intra-Frame Clashes"""
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
        return clashes

    def _get_frame_centers(self) -> List[Tuple[int, int, float, float]]:
        """Returns List of (row_idx, col_idx, x, y). Row 0 is North."""
        rows = 3
        cols = 5
        sx = self.geo.frame_spacing_x
        sy = self.geo.frame_spacing_y
        
        res = []
        for r in range(rows): # 0, 1, 2
            # Row 0 (North) -> Y = +sy
            # Row 1 (Center) -> Y = 0
            # Row 2 (South) -> Y = -sy
            y = (1 - r) * sy
            for c in range(cols): # 0, 1, 2, 3, 4
                x = (c - 2) * sx
                res.append((r, c, x, y))
        return res

    def _check_ray_intersection(self, origins: np.ndarray, directions: np.ndarray, 
                                blockers_min: np.ndarray, blockers_max: np.ndarray) -> np.ndarray:
        """
        Vectorized AABB Ray Intersection (Simplified for Panels).
        origins: (N, 3)
        directions: (N, 3) - Normalized Sun Vector
        blockers_min, blockers_max: (M, 3) - AABBs of M shading panels
        Returns: (N,) boolean - True if ray hits ANY blocker
        """
        # This is n_rays x m_blockers. Can be heavy.
        # We can accept boolean "hit any".
        # Expanded: (N, 1, 3) vs (1, M, 3)
        
        # Optimization: Only check blockers "upstream" towards sun.
        # For now, implemented as simple loop or broadcast if memory allows.
        # N ~ 16 (one frame active) vs M ~ 16 (one frame blocking). 256 checks. Fast.
        
        N = origins.shape[0]
        M = blockers_min.shape[0]
        
        O = origins[:, np.newaxis, :] # (N, 1, 3)
        D = directions[:, np.newaxis, :] # (N, 1, 3)
        
        # Slab method
        # t_min = (box_min - origin) / direction
        # t_max = (box_max - origin) / direction
        # We need to handle div by zero (use eps or numpy constraints)
        
        inv_D = 1.0 / (D + 1e-9)
        
        t0 = (blockers_min[np.newaxis, :, :] - O) * inv_D
        t1 = (blockers_max[np.newaxis, :, :] - O) * inv_D
        
        tmin = np.minimum(t0, t1)
        tmax = np.maximum(t0, t1)
        
        # Intersection interval is [max(tmin), min(tmax)]
        t_enter = np.max(tmin, axis=2) # (N, M)
        t_exit = np.min(tmax, axis=2)  # (N, M)
        
        # Hit if t_enter <= t_exit and t_exit > 0
        hits = (t_enter <= t_exit) & (t_exit > 0)
        
        return np.any(hits, axis=1)

    def run_year_power(self, year: int = 2024, frequency: str = '1H', excel_path: str = None):
        """Standard Year Run with Power Calculation"""
        if excel_path:
            self.load_dni_data(excel_path)
            
        print(f"1. Generating Solar Data ({frequency})...")
        times = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:59', freq=frequency, tz=self.location.tz)
        solpos = self.location.get_solarposition(times)
        # Filter Day
        day_mask = solpos['apparent_elevation'] > 0
        solpos = solpos[day_mask].copy()
        times_day = solpos.index
        
        dni = self.get_dni(times_day)
        az_all = solpos['azimuth'].values
        el_all = solpos['apparent_elevation'].values
        
        # Results containers
        total_power_kw = np.zeros(len(solpos))
        shadow_fractions = np.zeros(len(solpos)) # Avg shadow
        
        # Geometry Constants
        w, l, t = self.geo.width, self.geo.length, self.geo.thickness
        px, py = self.geo.pitch_x, self.geo.pitch_y
        sx, sy = self.geo.frame_spacing_x, self.geo.frame_spacing_y
        ox, oy, oz = self.geo.pivot_offset
        area = w * l
        
        # Pre-calc Frame Centers
        frame_layout = self._get_frame_centers() # List of (r, c, x, y)
        
        print(f"2. Calculating Metrics for {len(solpos)} timesteps...")
        
        # To optimize speed, we might want to loop but use simple math.
        # Or batch? 4000 steps is fine for a loop if simple.
        
        # Panel Local Coordinates (16 panels)
        # 4x4 grid.
        panel_local_origins = []
        for ix in range(4):
            for iy in range(4):
                panel_local_origins.append([ix*px, iy*py, 0])
        panel_local_origins = np.array(panel_local_origins) # (16, 3) Pivot Points
        
        # Panel Box (Relative to Pivot, unrotated) -> Pivot is at (0,0,-oz) + Offset?
        # SimulationEngine logic:
        # Neighbor defined as [pitch, 0, 0].
        # Pivot offset is vector from Pivot TO Geometric Center.
        # So Center = Pivot + Offset.
        # Box is Center +/- dims/2.
        center_local = np.array(self.geo.pivot_offset)
        box_min_local = center_local - np.array([w/2, l/2, t/2])
        box_max_local = center_local + np.array([w/2, l/2, t/2])
        
        # Frame Tilt Rotation Matrices (Pre-calc)
        # Row 0, 1, 2
        R_frame_tilts = []
        for r in range(3):
            tr = np.radians(self.frame_tilts[r])
            # Rot X: [1, 0, 0; 0, cos, -sin; 0, sin, cos]
            # Assming +Tilt raises North Edge? (Standard tilt South means facing South, so North edge up).
            # If Tilt is South-Facing, usually defined as positive.
            # Local Z up. Tilted means Z points South.
            # Rotate around X-axis.
            # R_x(alpha) * [0, 0, 1] = [0, -sina, cosa]. Points South (-y) if alpha > 0.
            # Verify coord system. X=East, Y=North.
            # If Y is North, South is -Y.
            # Facing South means Normal has -Y component.
            # So [0, -sin, cos]. Corresponds to R_x(alpha) with alpha > 0.
            c, s = np.cos(tr), np.sin(tr)
            Rf = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            R_frame_tilts.append(Rf)

        for i, (az, el, d_val) in enumerate(zip(az_all, el_all, dni)):
            if d_val <= 1.0:
                continue

            # 1. Global Sun Vector
            # Az is deg from North clockwise. El is deg from Horizon.
            # Z up, Y North from Plant Axis (Wait, Plant is rotated 5 deg)
            # We already handle frame_rot in _get_rotation_matrices for local checks.
            # Let's work in "Plant Aligned" Global Coordinates where Y is Plant North.
            # True North is -frame_rot degrees from Plant North?
            # Az_plant = Az_true - frame_rot.
            az_plant = az - self.frame_rot
            theta = np.radians(90 - az_plant) # Math angle from X
            phi = np.radians(el)
            
            # Sun Vector (towards Sun) in Plant Global
            sun_vec = np.array([
                np.cos(phi) * np.cos(theta),
                np.cos(phi) * np.sin(theta),
                np.sin(phi)
            ])
            sun_vec = sun_vec / np.linalg.norm(sun_vec)
            
            # 2. Panel Rotation (Tracking Sun)
            # We assume Perfect Tracking for Power Calc (unless in Stow, but basic calc first).
            # Local Panel Normal = Sun Vector (in Global Plant Frame? No, in Local Frame).
            # But since we track, Panel Normal aligns with Sun.
            # So Panel Orientation Matrix R_p aligns local Z (0,0,1) with Sun Vec.
            # However, for Shadowing, we need accurate BBox orientation.
            # R_p is same as R used in Clash: Rz(Az) @ Rx(El) (Local->Global or Global->Local?)
            
            # Reconstruct R_panel (Local to Global Plant)
            # Using logic from dashboard: c_world = (R @ c_shifted.T).T
            # So R is Local->Global.
            # R = Rz @ Rx
            az_rad = np.radians(az_plant) 
            theta_p = np.radians(90 - az_plant)
            c, s = np.cos(theta_p), np.sin(theta_p)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            
            tilt_rad = np.radians(90 - el)
            ct, st = np.cos(tilt_rad), np.sin(tilt_rad)
            Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
            
            R_panel = Rz @ Rx # Local (Pivot) -> Plant Global (untilted frame)
            
            # To be precise: If Frame is Tilted, Tracking angle is relative to Frame?
            # "Panels have 2-axis tracking relative to the frame."
            # If Frame Tilts, the Panel Tracking changes to compensate.
            # Result: Panel Normal still points to Sun (Global).
            # So R_panel (Local -> Global) is invariant of Frame Tilt if it tracks perfect Sun!
            # EXCEPT: The Frame Tilt changes the Pivot Point location in Global Space? No, Pivot is fixed on frame.
            # But the Panel mechanism boundaries might rotate. 
            # Let's assume the Panel Box Orientation in Global Space depends ONLY on Az/El (Sun position).
            # Because it tracks the sun.
            
            # 3. Inter-Frame Shadows
            # We iterate through Rows.
            # Row 0 (North): Never shaded by other frames (Sun is North-ish).
            # Shadows cast "behind" the sun.
            # In Southern Hemisphere (Koster, SA), Sun is North.
            # Shadows fall South.
            # So Row 0 shadows Row 1. Row 1 shadows Row 2.
            
            # We need to project Row r-1 shadows onto Row r.
            # Approximation:
            # We treat each 4x4 Frame as a "Cloud of Boxes".
            # For Row r (Target), we check if its panels are hit by rays from Row r-1 (Blocker).
            
            # Iterating 80 panels per timestep is expensive (250k steps).
            # Optimization: 
            # 1. Project Frame Center of Row r-1 along Sun Vec.
            # 2. If it is "close" to Frame Center of Row r, do detailed check.
            
            # Detailed check:
            # We pick Row 1 (Center) and Row 2 (South).
            # Blocker for Row 1 is Row 0.
            # Blocker for Row 2 is Row 1.
            
            frame_power_sum = 0
            
            # We assume all columns behave similarly? No, East/West early/late day effects.
            # But the plant axis is N-S. Columns are E-W.
            # Row shading is dominant.
            
            # Let's loop over all frames? Or just representative Row 0, Row 1, Row 2.
            # Since cols are 5, let's take Col 2 (Middle) as representative? 
            # Or assume Col 0 and Col 4 have edge effects?
            # User wants "whole factory". 
            # Cost: 15 frames * 16 panels * 8760 = 2 million checks.
            # AABB check is fast.
            # Python loop is slow.
            # Let's just do it for 3 representative Rows (Column 2).
            # And multiply by 5 columns?
            # The "5 deg rotation" makes columns slightly different.
            # Let's simulate Center Column (Col 2).
            # Frames: (0,2), (1,2), (2,2).
            
            # Frame 0 (North) - Unshaded.
            # Frame 1 (Center) - Shaded by Frame 0.
            # Frame 2 (South) - Shaded by Frame 1.
            
            # Calculate power for Frame 0 (16 panels). Perfect.
            p_frame_0_unit = area * d_val # 16 panels * ...
            
            # Calculate Shadow on Frame 1 from Frame 0.
            # Target: Frame 1 Panels. (Global positions)
            # Blocker: Frame 0 Panels. (Global positions)
            
            # Positions:
            # Frame Center Pos (Global):
            # Row r, Col c.
            # X = (c-2)*sx, Y = (1-r)*sy, Z = 0?
            # Z is modified by Ground Slope? Assumed flat.
            
            # Frame 0 Center: (0, sy, 0).
            # Frame 1 Center: (0, 0, 0).
            # Frame 2 Center: (0, -sy, 0).
            
            # Bounding Boxes of Blocker (Frame 0):
            # 16 panels.
            # Local pos in Frame: panel_local_origins.
            # Rotate by R_frame_tilt[0] (Row 0 tilt).
            # Add to Frame 0 Center.
            # Add R_panel @ Box_Local.
            
            blocker_boxes_min = []
            blocker_boxes_max = []
            
            # Helper to get Global BBoxes for a Frame
            def get_frame_bboxes(r_idx, c_idx, frame_tilt_R):
                f_y = (1 - r_idx) * sy
                f_x = (c_idx - 2) * sx
                f_center = np.array([f_x, f_y, 0])
                
                boxes_min = []
                boxes_max = []
                
                # Panel Pivot Centers in Global
                # p_local (16,3). Rotate by Frame Tilt.
                p_centers_frame = panel_local_origins @ frame_tilt_R.T
                p_centers_global = p_centers_frame + f_center
                
                # Panel Boxes in Global
                # Box is R_panel @ box_local_corners + P_center.
                # AABB of rotated box?
                # AABB min = Center + min(R @ corners)
                # We need AABB for ray check? Yes 
                # Corners of unrotated box:
                # 8 corners.
                # Just use Extent along axes.
                # Radius = max projection.
                # Let's project all 8 corners for accurate AABB.
                dx, dy, dz = w/2, l/2, t/2
                corners = np.array([
                    [dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],
                    [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz]
                ])
                # Rotate corners by R_panel
                corners_rot = corners @ R_panel.T # (8, 3)
                
                min_c = np.min(corners_rot, axis=0) + self.geo.pivot_offset @ R_panel.T # offset rotated too?
                # Wait, pivot_offset is in panel frame.
                # Center = Pivot + Offset.
                # Box is around Center.
                # Pivot is Global Point.
                # Box Global Verts = Pivot_Global + R_panel @ (Offset + Corner_Local_Rel_Center)
                # Wait, Box_min_local was relative to Pivot.
                # My `box_min_local` calc was: pivot_offset - dims/2.
                # So corners relative to Pivot are:
                # C_rel_p = pivot_offset + corner_rel_center.
                # Global Corner = Pivot_Global + R_panel @ C_rel_p.
                
                # Simpler:
                # Effective Radius for AABB?
                # Let's just compute the 8 corners in Global.
                
                # Refined:
                # For each panel k:
                #   Pivot_Ek = p_centers_global[k]
                #   Corners_Global = Pivot_Ek + (R_panel @ corners_rel_pivot.T).T
                #   bbox_min = min(Corners_Global)
                #   bbox_max ...
                
                # We need corners relative to pivot.
                # box_min_local IS relative to pivot.
                # But it's an AABB in local.
                # Vertices: combinations of min/max coords? No, box_min_local/max define the box in local.
                # Vertices of the box in local relative to pivot:
                v_local = []
                for vx in [box_min_local[0], box_max_local[0]]:
                    for vy in [box_min_local[1], box_max_local[1]]:
                        for vz in [box_min_local[2], box_max_local[2]]:
                            v_local.append([vx, vy, vz])
                v_local = np.array(v_local) # (8, 3)
                
                v_rotated = v_local @ R_panel.T # (8, 3)
                
                # Broadcasting for all 16 panels?
                # p_centers_global (16, 3).
                # All panels have same orientation R_panel.
                # So v_rotated is same for all.
                # Global BBox Min for panel k = p_centers_global[k] + min(v_rotated)
                # Global BBox Max for panel k = p_centers_global[k] + max(v_rotated)
                
                vmin = np.min(v_rotated, axis=0)
                vmax = np.max(v_rotated, axis=0)
                
                b_min = p_centers_global + vmin
                b_max = p_centers_global + vmax
                
                return b_min, b_max

            # Get Geometry for Row 0, 1, 2 (Center Column)
            b0_min, b0_max = get_frame_bboxes(0, 2, R_frame_tilts[0])
            b1_min, b1_max = get_frame_bboxes(1, 2, R_frame_tilts[1])
            b2_min, b2_max = get_frame_bboxes(2, 2, R_frame_tilts[2])
            
            # --- SHADING CHECKS ---
            
            # 1. Row 0 (North)
            # Unshaded (by other frames).
            f0_shad = 0.0
            
            # 2. Row 1 (Center)
            # Check against Row 0.
            # Targets: Centers of Panels in Row 1.
            # Approximation: Check Ray from Pivot 1.
            # Origins: Pivot points of Row 1.
            # p_centers_global for Row 1.
            p1_centers = panel_local_origins @ R_frame_tilts[1].T + np.array([0, 0, 0]) # Row 1 Center is (0,0,0)
            
            # Update p1_centers to be the actual geometric centers or target surface?
            # User: "amount of sunlight falling on the panels".
            # Check Center of Panel surface?
            # Surface center = Pivot + R_panel @ pivot_offset. (Global offset)
            p_offset_global = self.geo.pivot_offset @ R_panel.T
            targets_1 = p1_centers + p_offset_global
            
            hits_1 = self._check_ray_intersection(targets_1, np.tile(sun_vec, (16, 1)), b0_min, b0_max)
            f1_shad = np.mean(hits_1)
            
            # 3. Row 2 (South)
            # Check against Row 1.
            # Targets: Row 2.
            p2_centers = panel_local_origins @ R_frame_tilts[2].T + np.array([(2-2)*sx, (1-2)*sy, 0]) # x=0, y=-sy
            targets_2 = p2_centers + p_offset_global
            
            hits_2 = self._check_ray_intersection(targets_2, np.tile(sun_vec, (16, 1)), b1_min, b1_max)
            f2_shad = np.mean(hits_2)
            
            # --- TOTAL POWER ---
            # Sum for full plant.
            # Row 0: 5 frames. Shadow = 0.
            # Row 1: 5 frames. Shadow = f1_shad.
            # Row 2: 5 frames. Shadow = f2_shad.
            
            p_unit = 16 * area * d_val
            
            p_row_0 = 5 * p_unit # Clean
            p_row_1 = 5 * p_unit * (1.0 - f1_shad)
            p_row_2 = 5 * p_unit * (1.0 - f2_shad)
            
            total_power_kw[i] = (p_row_0 + p_row_1 + p_row_2) / 1000.0 # Watts to kW
            
            # Debug/Stats
            # if i % 1000 == 0:
            #    print(f"Step {i}: El={el:.1f}, Power={total_power_kw[i]:.2f} kW, Shad1={f1_shad:.2f}")

        solpos['Power_Total_kW'] = total_power_kw
        solpos['Power_Potential_kW'] = total_power_kw
        
        # Calculate Real Clashes (Intra-Frame)
        clash_result = self._calculate_clashes(az_all, el_all)
        solpos['Clash'] = clash_result
        
        # Apply Stow Loss (Checkerboard: 50% reduction during clash)
        # Power_Actual_kW is what goes to the grid.
        solpos['Power_Actual_kW'] = np.where(solpos['Clash'], solpos['Power_Potential_kW'] * 0.5, solpos['Power_Potential_kW'])
        
        # Rename for Dashboard Compatibility
        solpos.rename(columns={'azimuth': 'Azimuth', 'apparent_elevation': 'Elevation'}, inplace=True)
        
        # Ensure Time is a column
        solpos.index.name = 'Time'
        solpos.reset_index(inplace=True)
        return solpos, clash_result

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