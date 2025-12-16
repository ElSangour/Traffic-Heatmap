"""
Calibration Data Storage
Save and load calibration data (points, matrices, configuration)
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from calibration_system.config import CalibrationConfig, CameraConfig
from calibration_system.homography import HomographyCalculator


class CalibrationStorage:
    """
    Handles saving and loading calibration data
    Supports JSON format for easy editing and version control
    """
    
    def __init__(self, output_dir: str = "./calibration_data"):
        self.output_dir = output_dir
        self.homography_calc = HomographyCalculator()
        os.makedirs(output_dir, exist_ok=True)
    
    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize a string for use in filename
        Removes/replaces invalid characters
        """
        import re
        # Replace spaces with underscores
        name = name.replace(" ", "_")
        # Remove invalid filename characters
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        # Limit length
        name = name[:50]
        return name.lower()
    
    def save_calibration(
        self,
        config: CalibrationConfig,
        filename: str = None
    ) -> str:
        """
        Save complete calibration data to JSON
        
        Args:
            config: CalibrationConfig with all calibration data
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize store name for filename
            store_name_safe = self._sanitize_filename(config.store_name)
            filename = f"calibration_{store_name_safe}_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "store_name": config.store_name
            },
            "store": {
                "name": config.store_name,
                "plan_path": config.store_plan_path,
                "plan_width": config.store_plan_width,
                "plan_height": config.store_plan_height
            },
            "cameras": []
        }
        
        for cam in config.cameras:
            cam_data = {
                "camera_id": cam.camera_id,
                "name": cam.name,
                "rtsp_url": cam.rtsp_url,
                "enabled": cam.enabled,
                "calibration": {
                    "camera_points": cam.calibration_points_camera,
                    "plan_points": cam.calibration_points_plan,
                    "homography_matrix": cam.homography_matrix,
                    "is_calibrated": cam.is_calibrated()
                }
            }
            data["cameras"].append(cam_data)
        
        # Add calibration summary
        summary = config.get_calibration_summary()
        data["summary"] = summary
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[INFO] Calibration saved to: {filepath}")
        return filepath
    
    def load_calibration(self, filepath: str) -> Optional[CalibrationConfig]:
        """
        Load calibration data from JSON
        
        Args:
            filepath: Path to calibration JSON file
            
        Returns:
            CalibrationConfig or None if failed
        """
        if not os.path.exists(filepath):
            print(f"[ERROR] File not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            config = CalibrationConfig(
                store_name=data["store"]["name"],
                store_plan_path=data["store"]["plan_path"],
                store_plan_width=data["store"]["plan_width"],
                store_plan_height=data["store"]["plan_height"],
                num_cameras=0
            )
            config.cameras = []
            
            for cam_data in data["cameras"]:
                cam = CameraConfig(
                    camera_id=cam_data["camera_id"],
                    name=cam_data["name"],
                    rtsp_url=cam_data["rtsp_url"],
                    enabled=cam_data.get("enabled", True),
                    calibration_points_camera=cam_data["calibration"]["camera_points"],
                    calibration_points_plan=cam_data["calibration"]["plan_points"],
                    homography_matrix=cam_data["calibration"]["homography_matrix"]
                )
                config.cameras.append(cam)
            
            config.num_cameras = len(config.cameras)
            print(f"[INFO] Calibration loaded from: {filepath}")
            return config
            
        except Exception as e:
            print(f"[ERROR] Failed to load calibration: {e}")
            return None
    
    def export_homography_matrices(
        self,
        config: CalibrationConfig,
        filename: str = None
    ) -> str:
        """
        Export only the homography matrices (for use in detection pipeline)
        
        Returns:
            Path to exported file
        """
        if filename is None:
            # Sanitize store name for filename
            store_name_safe = self._sanitize_filename(config.store_name)
            filename = f"homography_matrices_{store_name_safe}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        matrices = {}
        for cam in config.cameras:
            if cam.is_calibrated() and cam.homography_matrix is not None:
                matrices[f"camera_{cam.camera_id}"] = {
                    "name": cam.name,
                    "matrix": cam.homography_matrix
                }
        
        data = {
            "store_name": config.store_name,
            "exported_at": datetime.now().isoformat(),
            "matrices": matrices
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[INFO] Homography matrices exported to: {filepath}")
        return filepath
    
    def load_homography_matrices(self, filepath: str) -> Dict[int, np.ndarray]:
        """
        Load homography matrices from export file
        
        Returns:
            Dictionary mapping camera_id to numpy matrix
        """
        if not os.path.exists(filepath):
            print(f"[ERROR] File not found: {filepath}")
            return {}
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        matrices = {}
        for key, value in data["matrices"].items():
            camera_id = int(key.replace("camera_", ""))
            matrices[camera_id] = np.array(value["matrix"], dtype=np.float64)
        
        return matrices
    
    def save_camera_snapshot(
        self,
        camera_id: int,
        frame: np.ndarray,
        suffix: str = ""
    ) -> str:
        """Save camera frame snapshot for reference"""
        import cv2
        
        snapshots_dir = os.path.join(self.output_dir, "snapshots")
        os.makedirs(snapshots_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_{camera_id}_{timestamp}{suffix}.jpg"
        filepath = os.path.join(snapshots_dir, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"[INFO] Snapshot saved: {filepath}")
        return filepath
    
    def list_calibrations(self) -> List[Dict]:
        """List all saved calibration files"""
        files = []
        for f in os.listdir(self.output_dir):
            if f.startswith("calibration_") and f.endswith(".json"):
                filepath = os.path.join(self.output_dir, f)
                stat = os.stat(filepath)
                files.append({
                    "filename": f,
                    "path": filepath,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        return sorted(files, key=lambda x: x["modified"], reverse=True)
    
    def get_latest_calibration(self) -> Optional[str]:
        """Get path to most recent calibration file"""
        files = self.list_calibrations()
        if files:
            return files[0]["path"]
        return None


if __name__ == "__main__":
    # Test storage
    storage = CalibrationStorage()
    
    # Create test config
    config = CalibrationConfig(
        store_name="Test Store",
        store_plan_path="./floor_plan.png",
        num_cameras=2
    )
    
    # Add calibration points
    config.cameras[0].calibration_points_camera = [[100, 100], [500, 100], [500, 400], [100, 400]]
    config.cameras[0].calibration_points_plan = [[50, 50], [350, 50], [350, 350], [50, 350]]
    config.cameras[0].homography_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    # Save
    filepath = storage.save_calibration(config)
    
    # List
    print("\nAvailable calibrations:")
    for f in storage.list_calibrations():
        print(f"  - {f['filename']}")
    
    # Load
    loaded = storage.load_calibration(filepath)
    print(f"\nLoaded config: {loaded.store_name} with {loaded.num_cameras} cameras")
