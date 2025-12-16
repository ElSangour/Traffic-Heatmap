"""
Homography Calculator
Computes homography matrices from point correspondences
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HomographyResult:
    """Result of homography calculation"""
    matrix: np.ndarray
    reprojection_error: float
    inliers_mask: Optional[np.ndarray] = None
    is_valid: bool = True
    error_message: str = ""


class HomographyCalculator:
    """
    Calculates homography matrices from point correspondences
    Transforms coordinates from camera view to floor plan
    """
    
    def __init__(self, min_points: int = 4):
        self.min_points = min_points
    
    def calculate_homography(
        self,
        camera_points: List[Tuple[float, float]],
        plan_points: List[Tuple[float, float]],
        method: int = cv2.RANSAC,
        ransac_threshold: float = 5.0
    ) -> HomographyResult:
        """
        Calculate homography matrix from point correspondences
        
        Args:
            camera_points: Points selected on camera image [(x1,y1), (x2,y2), ...]
            plan_points: Corresponding points on floor plan [(x1,y1), (x2,y2), ...]
            method: OpenCV method (cv2.RANSAC, cv2.LMEDS, or 0 for regular)
            ransac_threshold: RANSAC reprojection threshold
            
        Returns:
            HomographyResult with matrix and quality metrics
        """
        # Validate inputs
        if len(camera_points) < self.min_points:
            return HomographyResult(
                matrix=np.eye(3),
                reprojection_error=float('inf'),
                is_valid=False,
                error_message=f"Need at least {self.min_points} points, got {len(camera_points)}"
            )
        
        if len(camera_points) != len(plan_points):
            return HomographyResult(
                matrix=np.eye(3),
                reprojection_error=float('inf'),
                is_valid=False,
                error_message="Number of camera points must match plan points"
            )
        
        # Convert to numpy arrays
        src_pts = np.array(camera_points, dtype=np.float32)
        dst_pts = np.array(plan_points, dtype=np.float32)
        
        try:
            # Calculate homography
            if len(camera_points) == 4 and method == 0:
                # Exact solution for 4 points
                H = cv2.getPerspectiveTransform(src_pts, dst_pts)
                mask = None
            else:
                # RANSAC or LMEDS for more points
                H, mask = cv2.findHomography(src_pts, dst_pts, method, ransac_threshold)
            
            if H is None:
                return HomographyResult(
                    matrix=np.eye(3),
                    reprojection_error=float('inf'),
                    is_valid=False,
                    error_message="Homography calculation failed"
                )
            
            # Calculate reprojection error
            error = self._calculate_reprojection_error(src_pts, dst_pts, H)
            
            return HomographyResult(
                matrix=H,
                reprojection_error=error,
                inliers_mask=mask,
                is_valid=True
            )
            
        except Exception as e:
            return HomographyResult(
                matrix=np.eye(3),
                reprojection_error=float('inf'),
                is_valid=False,
                error_message=str(e)
            )
    
    def _calculate_reprojection_error(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        H: np.ndarray
    ) -> float:
        """Calculate mean reprojection error"""
        # Transform source points
        src_pts_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
        projected = (H @ src_pts_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        
        # Calculate error
        errors = np.sqrt(np.sum((projected - dst_pts) ** 2, axis=1))
        return float(np.mean(errors))
    
    def transform_point(
        self,
        point: Tuple[float, float],
        H: np.ndarray
    ) -> Tuple[float, float]:
        """
        Transform a single point using homography matrix
        
        Args:
            point: (x, y) in camera coordinates
            H: Homography matrix
            
        Returns:
            (x, y) in floor plan coordinates
        """
        pt = np.array([point[0], point[1], 1.0])
        transformed = H @ pt
        transformed = transformed / transformed[2]
        return (float(transformed[0]), float(transformed[1]))
    
    def transform_points(
        self,
        points: List[Tuple[float, float]],
        H: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Transform multiple points"""
        return [self.transform_point(p, H) for p in points]
    
    def inverse_transform_point(
        self,
        point: Tuple[float, float],
        H: np.ndarray
    ) -> Tuple[float, float]:
        """Transform from floor plan to camera coordinates"""
        H_inv = np.linalg.inv(H)
        return self.transform_point(point, H_inv)
    
    def validate_homography(self, H: np.ndarray) -> Tuple[bool, str]:
        """
        Validate homography matrix
        
        Returns:
            (is_valid, message)
        """
        # Check if matrix is 3x3
        if H.shape != (3, 3):
            return False, "Matrix must be 3x3"
        
        # Check determinant (should be non-zero)
        det = np.linalg.det(H)
        if abs(det) < 1e-10:
            return False, "Matrix is singular (determinant near zero)"
        
        # Check for reasonable values (no extreme scaling)
        if abs(det) > 1e6 or abs(det) < 1e-6:
            return False, f"Unusual determinant: {det:.2e}"
        
        # Check condition number (numerical stability)
        cond = np.linalg.cond(H)
        if cond > 1e6:
            return False, f"Poor conditioning: {cond:.2e}"
        
        return True, "Valid homography"
    
    def matrix_to_list(self, H: np.ndarray) -> List[List[float]]:
        """Convert numpy matrix to nested list for JSON serialization"""
        return H.tolist()
    
    def list_to_matrix(self, H_list: List[List[float]]) -> np.ndarray:
        """Convert nested list back to numpy matrix"""
        return np.array(H_list, dtype=np.float64)
    
    def create_test_visualization(
        self,
        camera_image: np.ndarray,
        plan_image: np.ndarray,
        camera_points: List[Tuple[float, float]],
        plan_points: List[Tuple[float, float]],
        H: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create visualization of calibration results
        
        Returns:
            (camera_image_with_grid, plan_image_with_projected_grid)
        """
        camera_vis = camera_image.copy()
        plan_vis = plan_image.copy()
        
        # Draw points on camera image
        for i, pt in enumerate(camera_points):
            cv2.circle(camera_vis, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
            cv2.putText(camera_vis, str(i+1), (int(pt[0])+10, int(pt[1])+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw points on plan image
        for i, pt in enumerate(plan_points):
            cv2.circle(plan_vis, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
            cv2.putText(plan_vis, str(i+1), (int(pt[0])+10, int(pt[1])+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw transformed points on plan (should match original points)
        transformed = self.transform_points(camera_points, H)
        for pt in transformed:
            cv2.circle(plan_vis, (int(pt[0]), int(pt[1])), 5, (255, 255, 0), -1)
        
        return camera_vis, plan_vis


if __name__ == "__main__":
    # Test homography calculation
    calculator = HomographyCalculator()
    
    # Test points (camera -> plan)
    camera_pts = [(100, 100), (500, 100), (500, 400), (100, 400)]
    plan_pts = [(50, 50), (350, 50), (350, 350), (50, 350)]
    
    result = calculator.calculate_homography(camera_pts, plan_pts)
    
    print(f"Valid: {result.is_valid}")
    print(f"Reprojection Error: {result.reprojection_error:.4f}")
    print(f"Matrix:\n{result.matrix}")
    
    # Test transformation
    test_point = (300, 250)
    transformed = calculator.transform_point(test_point, result.matrix)
    print(f"\nTest point {test_point} -> {transformed}")
    
    # Validate
    valid, msg = calculator.validate_homography(result.matrix)
    print(f"\nValidation: {valid} - {msg}")
