import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
import os

def preprocess_image( image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the image for analysis"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return gray, thresh


from typing import Optional, Tuple, List
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class CardCandidate:
    """Represents a potential card detection"""
    contour: np.ndarray
    area: float
    aspect_ratio: float
    rectangularity: float  # How close to a perfect rectangle
    score: float  # Overall quality score

class AdvancedCardDetector:
    """Advanced card detection with multiple strategies and validation"""
    
    def __init__(
        self,
        min_area_ratio: float = 0.1,
        max_area_ratio: float = 0.95,
        aspect_ratio_range: Tuple[float, float] = (1.2, 2.0),
        min_rectangularity: float = 0.85,
        use_edge_detection: bool = True,
        use_morphology: bool = True
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_ratio_range = aspect_ratio_range
        self.min_rectangularity = min_rectangularity
        self.use_edge_detection = use_edge_detection
        self.use_morphology = use_morphology
    
    def preprocess_advanced(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply multiple preprocessing strategies"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = []
        
        # Strategy 1: Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        results.append(adaptive_thresh)
        
        # Strategy 2: Otsu's thresholding with Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu_thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        results.append(otsu_thresh)
        
        # Strategy 3: Edge detection
        if self.use_edge_detection:
            edges = cv2.Canny(blurred, 50, 150)
            # Dilate edges to close gaps
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            results.append(edges_dilated)
        
        # Apply morphological operations to clean up
        if self.use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            for i, thresh in enumerate(results):
                # Close small holes
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                # Remove small noise
                opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
                results[i] = opened
        # If no results, use the entire grayscale image
        if not results:
                    results.append(gray)
                    
        return results
    
    def calculate_rectangularity(self, contour: np.ndarray) -> float:
        """
        Calculate how rectangular a contour is (0-1, 1 being perfect rectangle)
        """
        contour_area = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        box_area = rect[1][0] * rect[1][1]
        
        if box_area == 0:
            return 0.0
        
        return contour_area / box_area
    
    def calculate_corner_quality(self, approx: np.ndarray) -> float:
        """
        Evaluate the quality of corners (angles should be close to 90 degrees)
        """
        if len(approx) != 4:
            return 0.0
        
        angles = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % 4][0]
            p3 = approx[(i + 2) % 4][0]
            
            # Calculate angle using vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            angle = np.abs(np.arccos(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            ))
            angles.append(np.abs(angle - np.pi/2))  # Distance from 90 degrees
        
        # Lower is better, normalize to 0-1 scale
        avg_deviation = np.mean(angles)
        return max(0, 1 - (avg_deviation / (np.pi / 4)))
    
    def extract_card_candidates(
        self, 
        image: np.ndarray, 
        thresh: np.ndarray
    ) -> List[CardCandidate]:
        """Extract and score potential card contours"""
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.min_area_ratio
        max_area = image_area * self.max_area_ratio
        
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding rectangle
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            
            if width == 0 or height == 0:
                continue
            
            aspect_ratio = max(width, height) / min(width, height)
            
            # Filter by aspect ratio
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue
            
            # Calculate rectangularity
            rectangularity = self.calculate_rectangularity(contour)
            
            if rectangularity < self.min_rectangularity:
                continue
            
            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Try to get 4 corners
            if len(approx) != 4:
                # Try different epsilon values
                for eps_factor in [0.01, 0.03, 0.04, 0.05]:
                    epsilon = eps_factor * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) == 4:
                        break
                
                # Use minimum area rectangle as fallback
                if len(approx) != 4:
                    box = cv2.boxPoints(rect)
                    approx = box.astype(int)
            
            # Calculate corner quality
            corner_quality = self.calculate_corner_quality(approx)
            
            # Calculate composite score
            # Weighted by: area (40%), rectangularity (30%), corner quality (30%)
            area_score = min(area / image_area, 1.0)
            score = (
                0.4 * area_score +
                0.3 * rectangularity +
                0.3 * corner_quality
            )
            
            candidates.append(CardCandidate(
                contour=approx,
                area=area,
                aspect_ratio=aspect_ratio,
                rectangularity=rectangularity,
                score=score
            ))
        
        return candidates
    
    def detect_card_contour(
        self, 
        image: np.ndarray,
        debug: bool = False
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Detect the main card contour using advanced techniques.
        
        Args:
            image: Input image as numpy array
            debug: If True, return debug information
            
        Returns:
            Tuple of (contour, debug_info)
            - contour: Approximated contour with 4 corners
            - debug_info: Dictionary with detection details (if debug=True)
            
        Raises:
            ValueError: If no valid card contour is detected
        """
        # Get multiple preprocessed versions
        # Get multiple preprocessed versions
        thresh_images = self.preprocess_advanced(image)

        all_candidates = []

        # Extract candidates from each preprocessing strategy
        for thresh in thresh_images:
                    candidates = self.extract_card_candidates(image, thresh)
                    all_candidates.extend(candidates)

        # If no candidates found, use the entire image as the card
        if not all_candidates:
                    h, w = image.shape[:2]
                    # Create a contour representing the entire image
                    entire_image_contour = np.array([
                    [[0, 0]],
                    [[w-1, 0]],
                    [[w-1, h-1]],
                    [[0, h-1]]
                    ], dtype=np.int32)
                    
        image_area = h * w
        aspect_ratio = max(w, h) / min(w, h)
    
        # Create a candidate for the entire image with a moderate score
        all_candidates.append(CardCandidate(
                    contour=entire_image_contour,
                    area=image_area,
                    aspect_ratio=aspect_ratio,
                    rectangularity=1.0,  # Perfect rectangle
                    score=0.5  # Moderate score since it's a fallback
                    ))
        
        if not all_candidates:
            raise ValueError(
                "No valid card detected. Ensure the card is clearly visible "
                "and takes up at least 10% of the image."
            )
        
        # Sort by score and get the best candidate
        all_candidates.sort(key=lambda x: x.score, reverse=True)
        best_candidate = all_candidates[0]
        
        # Additional validation: check if multiple candidates are similar
        # This can indicate a more reliable detection
        similar_candidates = [
            c for c in all_candidates[:5]
            if abs(c.area - best_candidate.area) / best_candidate.area < 0.1
        ]
        
        confidence = len(similar_candidates) / 5.0  # Normalized confidence
        
        debug_info = None
        if debug:
            debug_info = {
                'total_candidates': len(all_candidates),
                'best_score': best_candidate.score,
                'area': best_candidate.area,
                'aspect_ratio': best_candidate.aspect_ratio,
                'rectangularity': best_candidate.rectangularity,
                'confidence': confidence,
                'all_candidates': all_candidates[:5]  # Top 5
            }
        
        # Order corners in consistent order (top-left, top-right, bottom-right, bottom-left)
        ordered_contour = self.order_corners(best_candidate.contour)
        
        return ordered_contour, debug_info
    
    def order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners in clockwise order starting from top-left"""
        corners = corners.reshape(4, 2)
        
        # Sort by y-coordinate
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        
        # Top two points
        top = sorted_by_y[:2]
        # Bottom two points
        bottom = sorted_by_y[2:]
        
        # Sort top points by x-coordinate
        top_sorted = top[np.argsort(top[:, 0])]
        # Sort bottom points by x-coordinate (reverse for clockwise)
        bottom_sorted = bottom[np.argsort(bottom[:, 0])[::-1]]
        
        # Combine: top-left, top-right, bottom-right, bottom-left
        ordered = np.vstack([top_sorted[0], top_sorted[1], bottom_sorted[0], bottom_sorted[1]])
        
        return ordered.reshape(4, 1, 2).astype(np.int32)


# Usage example:
def detect_card_contour(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Convenience function for advanced card detection.
    
    Args:
        image: Input image
        debug: Enable debug output
        
    Returns:
        Card contour with 4 ordered corners
    """
    detector = AdvancedCardDetector()
    contour, debug_info = detector.detect_card_contour(image, debug=debug)
    return contour


import cv2
import numpy as np
from typing import Tuple, Dict

def analyze_card_centering(image: np.ndarray, contour: np.ndarray) -> Tuple[float, Dict]:
    """
    Analyzes trading card centering using gradient-based border detection.
    
    Args:
        image: BGR image containing the card
        contour: Contour of the card region
    
    Returns:
        Tuple of (psa_score, details_dict)
        - psa_score: Float from 1-10 matching PSA grading standards
        - details_dict: Contains ratios, margins, and confidence metrics
    """
    # Extract card region
    x, y, w, h = cv2.boundingRect(contour)
    card_region = image[y:y+h, x:x+w]
    
    # Convert to grayscale and apply bilateral filter to preserve edges
    gray = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Calculate gradients to find border transitions
    grad_x = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    
    # Project gradients onto axes
    vertical_projection = np.sum(np.abs(grad_x), axis=0)
    horizontal_projection = np.sum(np.abs(grad_y), axis=1)
    
    # Smooth projections to reduce noise
    vertical_smooth = cv2.GaussianBlur(vertical_projection.reshape(-1, 1), (15, 1), 0).flatten()
    horizontal_smooth = cv2.GaussianBlur(horizontal_projection.reshape(-1, 1), (15, 1), 0).flatten()
    
    # Find border transitions in outer regions (first and last third)
    left_region = vertical_smooth[:w//3]
    right_region = vertical_smooth[2*w//3:]
    top_region = horizontal_smooth[:h//3]
    bottom_region = horizontal_smooth[2*h//3:]
    
    # Detect borders as peaks in gradient projections
    left_border = np.argmax(left_region)
    right_border = w - (len(right_region) - np.argmax(right_region))
    top_border = np.argmax(top_region)
    bottom_border = h - (len(bottom_region) - np.argmax(bottom_region))
    
    # Calculate margins
    left_margin = left_border
    right_margin = w - right_border
    top_margin = top_border
    bottom_margin = h - bottom_border
    
    # Validate margins are reasonable (3-35% of card dimension)
    min_margin = min(w, h) * 0.03
    max_margin = min(w, h) * 0.35
    
    margins_valid = (
        all(min_margin <= m <= max_margin for m in [left_margin, right_margin, top_margin, bottom_margin]) and
        max(left_margin, right_margin) / min(left_margin, right_margin) <= 3 and
        max(top_margin, bottom_margin) / min(top_margin, bottom_margin) <= 3
    )
    
    if not margins_valid:
        # Fallback: assume moderate centering issues
        left_margin = right_margin = w * 0.1
        top_margin = bottom_margin = h * 0.1
        confidence = 0.0
    else:
        # Calculate confidence based on peak strength
        peak_strength = (
            vertical_smooth[left_border] + 
            vertical_smooth[right_border] + 
            horizontal_smooth[top_border] + 
            horizontal_smooth[bottom_border]
        ) / 4
        mean_strength = (np.mean(vertical_smooth) + np.mean(horizontal_smooth)) / 2
        confidence = min(peak_strength / mean_strength * 50, 100) if mean_strength > 0 else 50.0
    
    # Calculate centering ratios (smaller/larger * 100)
    lr_ratio = min(left_margin, right_margin) / max(left_margin, right_margin) * 100
    tb_ratio = min(top_margin, bottom_margin) / max(top_margin, bottom_margin) * 100
    avg_ratio = (lr_ratio + tb_ratio) / 2
    
    # Convert to PSA score based on official standards
    # PSA 10: 55/45 or better (88.89%)
    # PSA 9: 60/40 or better (66.67%)
    # PSA 8: 65/35 or better (53.85%)
    # PSA 7: 70/30 or better (42.86%)
    psa_score  =avg_ratio
  
    # Return results
    details = {
        "psa_score": round(psa_score, 2),
        "lr_ratio": round(lr_ratio, 2),
        "tb_ratio": round(tb_ratio, 2),
        "overall_ratio": round(avg_ratio, 2),
        "left_margin": round(left_margin, 2),
        "right_margin": round(right_margin, 2),
        "top_margin": round(top_margin, 2),
        "bottom_margin": round(bottom_margin, 2),
        "confidence": round(confidence, 2)
    }
    
    return psa_score, details




import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from scipy import ndimage, stats
from scipy.signal import convolve2d
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CornerFeatures:
    """Container for extracted corner features"""
    # Geometric features
    corner_strength: float
    edge_sharpness: float
    curvature: float
    
    # Texture features
    lbp_uniformity: float
    gabor_response: float
    fractal_dimension: float
    
    # Damage indicators
    whitening_score: float
    crease_score: float
    roughness_score: float
    
    # Statistical features
    entropy: float
    homogeneity: float
    contrast: float
    
    # Frequency domain
    high_freq_energy: float
    spectral_falloff: float


class AdvancedCornerAnalyzer:
    """
    Advanced corner analysis using multi-scale feature extraction,
    frequency domain analysis, and statistical modeling
    OpenCV/NumPy only implementation
    """
    
    def __init__(
        self,
        corner_ratio: float = 0.15,
        use_deep_features: bool = True,
        anomaly_detection: bool = True,
        multi_scale: bool = True
    ):
        self.corner_ratio = corner_ratio
        self.use_deep_features = use_deep_features
        self.anomaly_detection = anomaly_detection
        self.multi_scale = multi_scale
        
        # Pre-compute Gabor filter bank for texture analysis
        self.gabor_filters = self._create_gabor_filterbank()
        
        # LBP parameters
        self.lbp_radius = 3
        self.lbp_points = 8 * self.lbp_radius
        
    def _create_gabor_filterbank(self) -> List[np.ndarray]:
        """Create Gabor filters at multiple orientations and frequencies"""
        filters = []
        ksize = 31
        
        # Multiple frequencies and orientations
        frequencies = [0.1, 0.2, 0.3]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for freq in frequencies:
            for theta in orientations:
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), 
                    sigma=5.0, 
                    theta=theta,
                    lambd=1.0/freq, 
                    gamma=0.5, 
                    psi=0
                )
                filters.append(kernel)
        
        return filters
    
    def extract_rotated_corners(
        self, 
        image: np.ndarray, 
        contour: np.ndarray
    ) -> List[np.ndarray]:
        """Extract corner regions with perspective correction"""
        corners = contour.reshape(4, 2).astype(np.float32)
        
        # Order corners consistently
        center = corners.mean(axis=0)
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        ordered_idx = np.argsort(angles)
        
        top_left_idx = ordered_idx[
            np.argmin(corners[ordered_idx][:, 0] + corners[ordered_idx][:, 1])
        ]
        roll_amount = np.where(ordered_idx == top_left_idx)[0][0]
        ordered_corners = corners[np.roll(ordered_idx, -roll_amount)]
        
        width = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
        height = np.linalg.norm(ordered_corners[3] - ordered_corners[0])
        corner_size = int(min(width, height) * self.corner_ratio)
        
        corner_regions = []
        
        for i, corner_point in enumerate(ordered_corners):
            prev_point = ordered_corners[(i - 1) % 4]
            next_point = ordered_corners[(i + 1) % 4]
            
            v1 = (next_point - corner_point) / np.linalg.norm(next_point - corner_point)
            v2 = (prev_point - corner_point) / np.linalg.norm(prev_point - corner_point)
            
            region_corners = np.array([
                corner_point,
                corner_point + v1 * corner_size,
                corner_point + (v1 + v2) * corner_size,
                corner_point + v2 * corner_size
            ], dtype=np.float32)
            
            dst_points = np.array([
                [0, 0],
                [corner_size, 0],
                [corner_size, corner_size],
                [0, corner_size]
            ], dtype=np.float32)
            
            matrix = cv2.getPerspectiveTransform(region_corners, dst_points)
            corner_region = cv2.warpPerspective(
                image, matrix, (corner_size, corner_size),
                flags=cv2.INTER_CUBIC
            )
            
            corner_regions.append(corner_region)
        
        return corner_regions
    
    def compute_local_binary_pattern(self, gray: np.ndarray) -> float:
        """
        Compute LBP uniformity score (custom implementation)
        Uniform patterns indicate smooth, undamaged surfaces
        """
        rows, cols = gray.shape
        lbp = np.zeros_like(gray)
        
        # Simple 8-neighbor LBP
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = gray[i, j]
                code = 0
                
                # 8 neighbors in circular order
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        # Calculate histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
        
        # Uniformity is inverse of entropy
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(256)
        
        uniformity = 1 - (entropy / max_entropy)
        return uniformity
    
    def compute_gabor_features(self, gray: np.ndarray) -> float:
        """
        Apply Gabor filter bank and compute texture response
        Strong responses at multiple orientations indicate damage
        """
        responses = []
        
        for kernel in self.gabor_filters:
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            responses.append(np.mean(np.abs(filtered)))
        
        # Standard deviation of responses (high = anisotropic damage)
        response_std = np.std(responses)
        response_mean = np.mean(responses)
        
        # Normalize: lower variation = better condition
        if response_mean > 0:
            coefficient_of_variation = response_std / response_mean
            score = max(0, 1 - coefficient_of_variation)
        else:
            score = 0.0
        
        return score
    
    def compute_fractal_dimension(self, gray: np.ndarray) -> float:
        """
        Compute fractal dimension using box-counting method
        Higher fractal dimension indicates rougher, more damaged surface
        """
        # Threshold image using Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Box counting
        scales = np.logspace(0.5, 3, num=10, base=2, dtype=int)
        counts = []
        
        for scale in scales:
            # Downsample
            if scale > 1:
                h, w = binary.shape
                new_h, new_w = h // scale, w // scale
                downsampled = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                downsampled = binary
            
            # Count occupied boxes
            count = np.sum(downsampled > 0)
            counts.append(count)
        
        # Linear fit in log-log space
        scales = scales[np.array(counts) > 0]
        counts = np.array(counts)[np.array(counts) > 0]
        
        if len(counts) > 1:
            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            fractal_dim = -coeffs[0]
        else:
            fractal_dim = 1.5
        
        # Normalize (typical range 1.0-2.0, lower is smoother)
        normalized = (fractal_dim - 1.0)
        return max(0, min(1, normalized))
    
    def compute_edge_sharpness(self, gray: np.ndarray) -> float:
        """
        Compute edge sharpness using gradient analysis
        Sharp, undamaged corners have well-defined edges
        """
        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Get edge pixels
        edges = magnitude > np.percentile(magnitude, 75)
        
        if np.sum(edges) == 0:
            return 0.0
        
        # Sharpness is measured by gradient magnitude at edges
        edge_strength = np.mean(magnitude[edges])
        
        # Normalize (typical range 0-100)
        sharpness = min(edge_strength / 50, 1.0)
        return sharpness
    
    def compute_corner_curvature(self, gray: np.ndarray) -> float:
        """
        Compute corner curvature using Harris corner detector
        Damaged corners have irregular curvature
        """
        # Harris corner detection
        gray_float = np.float32(gray)
        harris = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
        
        # Get maximum response (should be at corner)
        max_response = np.max(harris)
        
        # Normalize
        curvature_score = min(max_response / 1000, 1.0)
        return curvature_score
    
    def detect_creases(self, gray: np.ndarray) -> float:
        """
        Detect creases using Hough Line Transform
        Creases appear as linear features
        """
        # Edge detection
        edges = cv2.Canny(gray, 30, 100)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180,
            threshold=20,
            minLineLength=10,
            maxLineGap=5
        )
        
        if lines is None:
            return 0.0
        
        # Count and measure lines (more lines = more damage)
        num_lines = len(lines)
        
        # Normalize (0-10 lines expected in damaged corners)
        crease_score = min(num_lines / 10, 1.0)
        return crease_score
    
    def compute_glcm_features(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Compute Gray Level Co-occurrence Matrix features
        Provides texture characteristics
        """
        # Normalize to 0-255 and convert to uint8
        normalized = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-6) * 255)
        normalized = normalized.astype(np.uint8)
        
        # Reduce levels for efficiency
        levels = 32
        quantized = (normalized // (256 // levels)).astype(np.uint8)
        
        # Compute GLCM (simple version - one direction)
        glcm = np.zeros((levels, levels))
        
        for i in range(quantized.shape[0] - 1):
            for j in range(quantized.shape[1] - 1):
                glcm[quantized[i, j], quantized[i, j+1]] += 1
                glcm[quantized[i, j], quantized[i+1, j]] += 1
        
        # Normalize
        glcm = glcm / (glcm.sum() + 1e-6)
        
        # Compute features
        # Contrast
        i, j = np.ogrid[:levels, :levels]
        contrast = np.sum(glcm * (i - j)**2)
        
        # Homogeneity
        homogeneity = np.sum(glcm / (1 + (i - j)**2))
        
        # Entropy
        entropy_val = -np.sum(glcm[glcm > 0] * np.log2(glcm[glcm > 0]))
        
        return {
            'contrast': contrast / 100,  # Normalize
            'homogeneity': homogeneity,
            'entropy': entropy_val / 10  # Normalize
        }
    
    def compute_frequency_features(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Analyze frequency domain for damage detection
        Damaged areas have different spectral characteristics
        """
        # FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Radial frequency distribution
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create radial mask
        y, x = np.ogrid[:rows, :cols]
        radius = np.sqrt((x - ccol)**2 + (y - crow)**2)
        
        # High frequency energy (damage shows as high frequency)
        high_freq_mask = radius > (min(rows, cols) // 4)
        high_freq_energy = np.sum(magnitude[high_freq_mask])
        total_energy = np.sum(magnitude)
        
        high_freq_ratio = high_freq_energy / (total_energy + 1e-6)
        
        # Spectral falloff (smooth surfaces have rapid falloff)
        radii = np.arange(0, min(rows, cols) // 2, 5)
        spectrum = []
        
        for r in radii:
            mask = (radius >= r) & (radius < r + 5)
            spectrum.append(np.mean(magnitude[mask]))
        
        # Compute falloff rate
        spectrum = np.array(spectrum)
        if len(spectrum) > 1:
            # Fit exponential decay
            log_spectrum = np.log(spectrum + 1)
            falloff_rate = -np.polyfit(radii[:len(spectrum)], log_spectrum, 1)[0]
        else:
            falloff_rate = 0.0
        
        return {
            'high_freq_energy': min(high_freq_ratio * 10, 1.0),
            'spectral_falloff': min(falloff_rate, 1.0)
        }
    
    def compute_whitening_advanced(self, image: np.ndarray, gray: np.ndarray) -> float:
        """
        Advanced whitening detection using LAB color space and adaptive methods
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Adaptive threshold based on local statistics
        mean_l = np.mean(l_channel)
        std_l = np.std(l_channel)
        
        # Whitening appears as high L values
        whitening_threshold = mean_l + 1.5 * std_l
        whitening_mask = l_channel > whitening_threshold
        
        whitening_ratio = np.sum(whitening_mask) / whitening_mask.size
        
        # Also check for low saturation (white areas are desaturated)
        a_channel = lab[:, :, 1].astype(np.float32)
        b_channel = lab[:, :, 2].astype(np.float32)
        saturation = np.sqrt(a_channel**2 + b_channel**2)
        
        low_sat_threshold = np.percentile(saturation, 25)
        desaturated = saturation < low_sat_threshold
        
        # Combined whitening score
        combined_whitening = np.sum(whitening_mask & desaturated) / whitening_mask.size
        
        return combined_whitening
    
    def extract_deep_features(self, corner_region: np.ndarray) -> CornerFeatures:
        """
        Extract comprehensive feature set from corner region
        """
        gray = cv2.cvtColor(corner_region, cv2.COLOR_BGR2GRAY)
        
        # Geometric features
        corner_strength = self.compute_corner_curvature(gray)
        edge_sharpness = self.compute_edge_sharpness(gray)
        curvature = corner_strength  # Reuse for simplicity
        
        # Texture features
        lbp_uniformity = self.compute_local_binary_pattern(gray)
        gabor_response = self.compute_gabor_features(gray)
        fractal_dimension = self.compute_fractal_dimension(gray)
        
        # Damage indicators
        whitening_score = self.compute_whitening_advanced(corner_region, gray)
        crease_score = self.detect_creases(gray)
        roughness_score = 1 - lbp_uniformity  # Inverse of uniformity
        
        # Statistical features
        glcm_features = self.compute_glcm_features(gray)
        
        # Frequency domain
        freq_features = self.compute_frequency_features(gray)
        
        return CornerFeatures(
            corner_strength=corner_strength,
            edge_sharpness=edge_sharpness,
            curvature=curvature,
            lbp_uniformity=lbp_uniformity,
            gabor_response=gabor_response,
            fractal_dimension=fractal_dimension,
            whitening_score=whitening_score,
            crease_score=crease_score,
            roughness_score=roughness_score,
            entropy=glcm_features['entropy'],
            homogeneity=glcm_features['homogeneity'],
            contrast=glcm_features['contrast'],
            high_freq_energy=freq_features['high_freq_energy'],
            spectral_falloff=freq_features['spectral_falloff']
        )
    
    def compute_corner_score(self, features: CornerFeatures) -> float:
        """
        Compute overall corner score from features using weighted model
        """
        # Feature weights (empirically determined)
        weights = {
            'whitening': 0.25,
            'edge_sharpness': 0.15,
            'corner_strength': 0.15,
            'lbp_uniformity': 0.10,
            'gabor': 0.08,
            'crease': 0.12,
            'fractal': 0.05,
            'homogeneity': 0.05,
            'spectral': 0.05
        }
        
        # Compute weighted score (higher is better)
        score = (
            (1 - features.whitening_score) * weights['whitening'] * 100 +
            features.edge_sharpness * weights['edge_sharpness'] * 100 +
            features.corner_strength * weights['corner_strength'] * 100 +
            features.lbp_uniformity * weights['lbp_uniformity'] * 100 +
            features.gabor_response * weights['gabor'] * 100 +
            (1 - features.crease_score) * weights['crease'] * 100 +
            (1 - features.fractal_dimension) * weights['fractal'] * 100 +
            features.homogeneity * weights['homogeneity'] * 100 +
            features.spectral_falloff * weights['spectral'] * 100
        )
        
        return score
    
    def detect_anomalies_simple(self, corner_scores: np.ndarray) -> np.ndarray:
        """
        Simple anomaly detection using statistical outlier detection
        (replacement for Isolation Forest)
        """
        if len(corner_scores) < 4:
            return np.zeros(len(corner_scores))
        
        mean = np.mean(corner_scores)
        std = np.std(corner_scores)
        
        # Mark as anomaly if more than 1.5 standard deviations from mean
        z_scores = np.abs((corner_scores - mean) / (std + 1e-6))
        anomalies = (z_scores > 1.5).astype(float)
        
        return anomalies
    
    def analyze_corners(
        self,
        image: np.ndarray,
        contour: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Main analysis function using advanced methods
        
        Returns:
            Tuple of (overall_score, detailed_results)
        """
        if contour.shape[0] != 4:
            raise ValueError(f"Contour must have 4 points, got {contour.shape[0]}")
        
        # Extract corner regions
        corner_regions = self.extract_rotated_corners(image, contour)
        
        # Extract features for each corner
        all_features = []
        corner_scores = []
        corner_results = []
        
        corner_names = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
        
        for i, (region, name) in enumerate(zip(corner_regions, corner_names)):
            # Extract deep features
            features = self.extract_deep_features(region)
            all_features.append(features)
            
            # Compute score
            score = self.compute_corner_score(features)
            corner_scores.append(score)
            
            # Store results
            corner_results.append({
                'name': name,
                'score': round(score, 2),
                'features': {
                    'whitening': round(features.whitening_score, 4),
                    'edge_sharpness': round(features.edge_sharpness, 4),
                    'corner_strength': round(features.corner_strength, 4),
                    'lbp_uniformity': round(features.lbp_uniformity, 4),
                    'gabor_response': round(features.gabor_response, 4),
                    'crease_score': round(features.crease_score, 4),
                    'fractal_dimension': round(features.fractal_dimension, 4),
                    'homogeneity': round(features.homogeneity, 4),
                    'high_freq_energy': round(features.high_freq_energy, 4)
                }
            })
        
        # Anomaly detection
        corner_scores_array = np.array(corner_scores)
        if self.anomaly_detection:
            anomaly_scores = self.detect_anomalies_simple(corner_scores_array)
            for i, (result, is_anomaly) in enumerate(zip(corner_results, anomaly_scores)):
                result['is_anomalous'] = bool(is_anomaly)
        
        # Compute overall score
        overall_score = np.mean(corner_scores_array)
        
        # Statistical analysis
        worst_corner_idx = np.argmin(corner_scores_array)
        best_corner_idx = np.argmax(corner_scores_array)
        
        # Confidence interval (95%)
        confidence_interval = stats.t.interval(
            0.95, 
            len(corner_scores_array) - 1,
            loc=overall_score,
            scale=stats.sem(corner_scores_array)
        )
        
        details = {
            'overall_score': round(overall_score, 2),
            'confidence_interval': [round(confidence_interval[0], 2), round(confidence_interval[1], 2)],
            'corners': corner_results,
            'worst_corner': corner_names[worst_corner_idx],
            'worst_corner_score': round(corner_scores_array[worst_corner_idx], 2),
            'best_corner': corner_names[best_corner_idx],
            'best_corner_score': round(corner_scores_array[best_corner_idx], 2),
            'score_std': round(np.std(corner_scores_array), 2),
            'score_range': round(np.ptp(corner_scores_array), 2),
            'grade': self._get_grade(overall_score),
            'consistency': self._get_consistency_rating(np.std(corner_scores_array))
        }
        
        return overall_score, details
    
    def _get_grade(self, score: float) -> str:
        """Convert score to grade"""
        if score >= 95:
            return "Gem Mint (10)"
        elif score >= 90:
            return "Mint (9)"
        elif score >= 85:
            return "Near Mint-Mint (8.5)"
        elif score >= 80:
            return "Near Mint (8)"
        elif score >= 75:
            return "Excellent-Mint (7)"
        elif score >= 70:
            return "Excellent (6)"
        elif score >= 60:
            return "Good (5)"
        elif score >= 50:
            return "Fair (4)"
        else:
            return "Poor (1-3)"
    
    def _get_consistency_rating(self, std: float) -> str:
        """Rate corner consistency"""
        if std < 3:
            return "Excellent - All corners similar"
        elif std < 6:
            return "Good - Minor variation"
        elif std < 10:
            return "Fair - Noticeable variation"
        else:
            return "Poor - Significant variation"


# Convenience function
def analyze_corners_advanced(
    image: np.ndarray,
    contour: np.ndarray,
    verbose: bool = False
) -> Tuple[float, Dict]:
    """
    Advanced corner analysis with comprehensive feature extraction
    
    Args:
        image: Input BGR image
        contour: Card contour (4 points)
        verbose: Print detailed analysis
        
    Returns:
        Tuple of (overall_score, detailed_results)
    """
    analyzer = AdvancedCornerAnalyzer(
        use_deep_features=True,
        anomaly_detection=True,
        multi_scale=True
    )
    
    score, details = analyzer.analyze_corners(image, contour)

    return score, details



import numpy as np
import cv2
from scipy import ndimage
from dataclasses import dataclass
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle
from matplotlib.gridspec import GridSpec


@dataclass
class EdgeConfig:
    """Configuration for edge analysis parameters"""
    edge_sample_width: int = 5  # Pixels to sample from true edge
    white_threshold_low: int = 200  # Lower bound for whitening detection
    white_threshold_high: int = 235  # Upper bound for severe whitening
    
    # Scoring weights (corners removed)
    smoothness_weight: float = 0.40
    whitening_weight: float = 0.35
    continuity_weight: float = 0.25


class AdvancedEdgeAnalyzer:
    """
    Advanced edge quality analysis for trading cards - EDGE FOCUSED
    
    Features:
    - True edge extraction using contour normals
    - Adaptive lighting normalization
    - Multi-threshold whitening detection
    - Configurable parameters for different card types
    """
    
    def __init__(self, config: EdgeConfig = None):
        self.config = config or EdgeConfig()
    
    def analyze_edges(self, image: np.ndarray, contour: np.ndarray) -> Tuple[float, Dict]:
        """
        Main analysis function - EDGES ONLY
        
        Args:
            image: BGR image
            contour: Card contour from cv2.findContours
            
        Returns:
            (average_score, details_dict)
        """
        # Normalize lighting first
        normalized_image = self._normalize_lighting(image)
        
        # Extract true edge samples
        edge_samples = self._extract_edge_samples(normalized_image, contour)
        
        # Analyze each edge
        edge_scores = {}
        edge_metrics = {}
        
        for edge_name, samples in edge_samples.items():
            if samples is None or len(samples) == 0:
                continue
                
            score, metrics = self._analyze_edge_region(samples)
            edge_scores[edge_name] = score
            edge_metrics[edge_name] = metrics
        
        # Compute average score (edges only)
        average_score = np.mean(list(edge_scores.values())) if edge_scores else 0
        
        return round(average_score, 2), {
            "edge_scores": edge_scores,
            "average": round(average_score, 2),
            "edge_metrics": edge_metrics
        }
    
    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to normalize lighting across the card"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_normalized = clahe.apply(l)
        
        normalized = cv2.merge([l_normalized, a, b])
        return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)
    
    def _extract_edge_samples(self, image: np.ndarray, contour: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract true edge samples by sampling perpendicular to contour
        
        Returns dict with 'top', 'bottom', 'left', 'right' edge samples
        """
        # Approximate contour to get cleaner edges
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 4:
            return {}
        
        # Get the four corners (assuming rectangular card)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(box)
        
        edge_samples = {}
        edge_names = ['top', 'right', 'bottom', 'left']
        
        for i, edge_name in enumerate(edge_names):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            
            samples = self._sample_along_edge(image, p1, p2)
            edge_samples[edge_name] = samples
        
        return edge_samples
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left"""
        # Find center
        center = corners.mean(axis=0)
        
        # Sort by angle from center
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        
        # Start from top-left (angle around -135 degrees)
        start_idx = np.argmin(np.abs(angles - (-3 * np.pi / 4)))
        ordered = np.roll(sorted_indices, -np.where(sorted_indices == start_idx)[0][0])
        
        return corners[ordered]
    
    def _sample_along_edge(self, image: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Sample pixels along an edge, perpendicular to the edge direction"""
        edge_vector = p2 - p1
        edge_length = np.linalg.norm(edge_vector)
        edge_direction = edge_vector / edge_length
        
        # Perpendicular direction (inward)
        perp_direction = np.array([-edge_direction[1], edge_direction[0]])
        
        samples = []
        num_samples = int(edge_length)
        
        for i in range(num_samples):
            # Point along the edge
            t = i / num_samples
            edge_point = p1 + t * edge_vector
            
            # Sample perpendicular to edge (inward)
            for d in range(self.config.edge_sample_width):
                sample_point = edge_point + d * perp_direction
                x, y = int(sample_point[0]), int(sample_point[1])
                
                if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                    samples.append(image[y, x])
        
        return np.array(samples) if samples else np.array([])
    
    def _analyze_edge_region(self, samples: np.ndarray) -> Tuple[float, Dict]:
        """Analyze edge quality from sample pixels"""
        if len(samples) == 0:
            return 0.0, {}
        
        gray_samples = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()
        
        # 1. Texture roughness (using standard deviation as proxy)
        texture_std = np.std(gray_samples)
        smoothness_score = np.clip(100 - texture_std * 2, 0, 100)
        
        # 2. Multi-threshold whitening detection
        light_whitening = np.sum(gray_samples > self.config.white_threshold_low) / len(gray_samples)
        severe_whitening = np.sum(gray_samples > self.config.white_threshold_high) / len(gray_samples)
        
        # Penalize severe whitening more heavily
        whitening_penalty = light_whitening * 200 + severe_whitening * 400
        whitening_score = np.clip(100 - whitening_penalty, 0, 100)
        
        # 3. Edge continuity (detect breaks/chips)
        gradient = np.abs(np.diff(gray_samples))
        discontinuity = np.sum(gradient > 20) / len(gradient)  # Sharp changes indicate chips
        continuity_score = np.clip(100 - discontinuity * 500, 0, 100)
        
        # Weighted score
        final_score = (
            self.config.smoothness_weight * smoothness_score +
            self.config.whitening_weight * whitening_score +
            self.config.continuity_weight * continuity_score
        )
        
        metrics = {
            "texture_std": round(float(texture_std), 2),
            "light_whitening_ratio": round(float(light_whitening), 4),
            "severe_whitening_ratio": round(float(severe_whitening), 4),
            "discontinuity": round(float(discontinuity), 4),
            "smoothness_score": round(float(smoothness_score), 2),
            "whitening_score": round(float(whitening_score), 2),
            "continuity_score": round(float(continuity_score), 2)
        }
        
        return final_score, metrics


# Visualization using Matplotlib

def create_visualization(image: np.ndarray, contour: np.ndarray, 
                        score: float, details: Dict) -> plt.Figure:
    """
    Create comprehensive edge visualization using matplotlib
    
    Args:
        image: Original card image (BGR)
        contour: Card contour
        score: Overall edge score
        details: Analysis details from analyze_edges()
        
    Returns:
        matplotlib Figure
    """
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Main image with edge annotations
    ax_main = fig.add_subplot(gs[:, :2])
    ax_main.imshow(image_rgb)
    ax_main.axis('off')
    ax_main.set_title('Edge Quality Analysis (Edges Only)', fontsize=16, fontweight='bold', pad=20)
    
    # Draw contour
    contour_points = contour.squeeze()
    if len(contour_points.shape) == 2:
        ax_main.plot(np.append(contour_points[:, 0], contour_points[0, 0]),
                    np.append(contour_points[:, 1], contour_points[0, 1]),
                    'g-', linewidth=3, label='Card Boundary', alpha=0.7)
    
    # Get corners for edge positioning
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    
    # Draw edge quality indicators at midpoints
    h, w = image.shape[:2]
    edge_positions = {
        'top': (w // 2, 30),
        'bottom': (w // 2, h - 30),
        'left': (30, h // 2),
        'right': (w - 30, h // 2)
    }
    
    for edge_name, (ex, ey) in edge_positions.items():
        if edge_name in details.get('edge_scores', {}):
            edge_score_val = details['edge_scores'][edge_name]
            color = _score_to_color(edge_score_val)
            
            # Larger circles for edge indicators
            circle = Circle((ex, ey), 20, color=color, alpha=0.8, ec='black', linewidth=3)
            ax_main.add_patch(circle)
            
            # Add edge label
            label_offset = {'top': -35, 'bottom': 35, 'left': -35, 'right': 35}
            ax_main.text(ex, ey + label_offset[edge_name], edge_name.upper(), 
                        ha='center', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    # Overall Score Panel
    ax_score = fig.add_subplot(gs[0, 2])
    ax_score.axis('off')
    
    score_color = _score_to_color(score)
    grade = _score_to_grade(score)
    
    ax_score.text(0.5, 0.7, 'OVERALL EDGE QUALITY', ha='center', fontsize=12, 
                 fontweight='bold', transform=ax_score.transAxes)
    ax_score.text(0.5, 0.4, f'{score:.1f}', ha='center', fontsize=40, 
                 fontweight='bold', color=score_color, transform=ax_score.transAxes)
    ax_score.text(0.5, 0.2, f'Est. Grade: {grade}', ha='center', fontsize=10,
                 transform=ax_score.transAxes, style='italic')
    
    # Edge Scores Panel
    ax_edges = fig.add_subplot(gs[1, 2])
    ax_edges.axis('off')
    ax_edges.text(0.5, 0.95, 'INDIVIDUAL EDGE SCORES', ha='center', fontsize=11, 
                 fontweight='bold', transform=ax_edges.transAxes)
    
    y_pos = 0.75
    for edge_name, edge_score_val in details.get('edge_scores', {}).items():
        color = _score_to_color(edge_score_val)
        
        # Background bar
        rect_bg = Rectangle((0.1, y_pos - 0.05), 0.5, 0.08, 
                           facecolor='lightgray', transform=ax_edges.transAxes)
        ax_edges.add_patch(rect_bg)
        
        # Score bar
        rect_score = Rectangle((0.1, y_pos - 0.05), 0.5 * (edge_score_val / 100), 0.08,
                              facecolor=color, transform=ax_edges.transAxes)
        ax_edges.add_patch(rect_score)
        
        ax_edges.text(0.65, y_pos, f'{edge_name.capitalize()}: {edge_score_val:.1f}',
                     va='center', fontsize=10, fontweight='bold', transform=ax_edges.transAxes)
        y_pos -= 0.18
    
    # Detailed Metrics Panel
    ax_metrics = fig.add_subplot(gs[2:, 2])
    ax_metrics.axis('off')
    ax_metrics.text(0.5, 0.95, 'EDGE QUALITY METRICS', ha='center', fontsize=11,
                   fontweight='bold', transform=ax_metrics.transAxes)
    
    # Display metrics for each edge
    y_pos = 0.85
    for edge_name, metrics in details.get('edge_metrics', {}).items():
        ax_metrics.text(0.1, y_pos, f'{edge_name.upper()} Edge:', 
                       fontweight='bold', fontsize=10, transform=ax_metrics.transAxes)
        y_pos -= 0.08
        
        metric_items = [
            f"  • Smoothness: {metrics['smoothness_score']:.1f}/100",
            f"  • Whitening: {metrics['whitening_score']:.1f}/100 ({metrics['light_whitening_ratio']*100:.1f}% affected)",
            f"  • Continuity: {metrics['continuity_score']:.1f}/100 ({metrics['discontinuity']*100:.1f}% breaks)",
            f"  • Texture Std: {metrics['texture_std']:.2f}"
        ]
        
        for item in metric_items:
            ax_metrics.text(0.1, y_pos, item, fontsize=8, 
                           family='monospace', transform=ax_metrics.transAxes)
            y_pos -= 0.06
        
        y_pos -= 0.04
        
        if y_pos < 0.1:
            break
    
    plt.tight_layout()
    return fig


def create_heatmap(image: np.ndarray, contour: np.ndarray, 
                   details: Dict, config: EdgeConfig = None) -> plt.Figure:
    """
    Create a heatmap overlay showing edge quality along the card edges
    
    Args:
        image: Original card image (BGR)
        contour: Card contour
        details: Analysis details
        config: EdgeConfig
        
    Returns:
        matplotlib Figure with heatmap
    """
    if config is None:
        config = EdgeConfig()
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = np.zeros_like(image_rgb)
    
    # Get corners
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    
    # Draw edge heatmaps with thicker lines
    edge_thickness = 25
    edge_names = ['top', 'right', 'bottom', 'left']
    
    for i in range(len(box)):
        p1 = tuple(box[i])
        p2 = tuple(box[(i + 1) % len(box)])
        edge_name = edge_names[i]
        
        if edge_name in details.get('edge_scores', {}):
            edge_score = details['edge_scores'][edge_name]
            color_rgb = _score_to_color_rgb(edge_score)
            cv2.line(overlay, p1, p2, color_rgb, edge_thickness)
    
    # Blend
    alpha = 0.5
    blended = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(blended)
    ax.axis('off')
    ax.set_title('Edge Quality Heatmap', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=_score_to_color(95), label='Excellent (90-100)'),
        mpatches.Patch(color=_score_to_color(85), label='Mint (80-90)'),
        mpatches.Patch(color=_score_to_color(75), label='Near Mint (70-80)'),
        mpatches.Patch(color=_score_to_color(65), label='Good (60-70)'),
        mpatches.Patch(color=_score_to_color(45), label='Poor (<60)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, fontsize=11)
    
    plt.tight_layout()
    return fig


def create_edge_profile_chart(details: Dict) -> plt.Figure:
    """
    Create a radar/polar chart showing edge quality profile
    
    Args:
        details: Analysis details from analyze_edges()
        
    Returns:
        matplotlib Figure with polar chart
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), 
                                    subplot_kw=dict(projection='polar'))
    
    # Get edge scores
    edge_names = ['top', 'right', 'bottom', 'left']
    scores = [details['edge_scores'].get(name, 0) for name in edge_names]
    
    # Add first value to close the circle
    scores_closed = scores + [scores[0]]
    angles = np.linspace(0, 2 * np.pi, len(edge_names), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]
    
    # Plot 1: Overall edge scores
    ax1.plot(angles_closed, scores_closed, 'o-', linewidth=3, color='#2E86AB', markersize=10)
    ax1.fill(angles_closed, scores_closed, alpha=0.25, color='#2E86AB')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(angles)
    ax1.set_xticklabels([name.upper() for name in edge_names], fontsize=11, fontweight='bold')
    ax1.set_yticks([25, 50, 75, 100])
    ax1.set_yticklabels(['25', '50', '75', '100'])
    ax1.set_title('Edge Scores Profile', fontsize=13, fontweight='bold', pad=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Component scores (smoothness, whitening, continuity)
    if edge_names[0] in details.get('edge_metrics', {}):
        metrics = details['edge_metrics'][edge_names[0]]  # Use top edge as example
        component_names = ['Smoothness', 'Whitening', 'Continuity']
        component_scores = [
            metrics['smoothness_score'],
            metrics['whitening_score'],
            metrics['continuity_score']
        ]
        component_scores_closed = component_scores + [component_scores[0]]
        angles_comp = np.linspace(0, 2 * np.pi, len(component_names), endpoint=False).tolist()
        angles_comp_closed = angles_comp + [angles_comp[0]]
        
        ax2.plot(angles_comp_closed, component_scores_closed, 'o-', 
                linewidth=3, color='#A23B72', markersize=10)
        ax2.fill(angles_comp_closed, component_scores_closed, alpha=0.25, color='#A23B72')
        ax2.set_ylim(0, 100)
        ax2.set_xticks(angles_comp)
        ax2.set_xticklabels(component_names, fontsize=11, fontweight='bold')
        ax2.set_yticks([25, 50, 75, 100])
        ax2.set_yticklabels(['25', '50', '75', '100'])
        ax2.set_title('Quality Components (Top Edge)', fontsize=13, fontweight='bold', pad=20)
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def _score_to_color(score: float) -> str:
    """Convert score to hex color for matplotlib"""
    if score >= 90:
        return '#00C800'  # Green
    elif score >= 80:
        return '#64DC64'  # Light green
    elif score >= 70:
        return '#C8FFC8'  # Yellow-green
    elif score >= 60:
        return '#FFFF00'  # Yellow
    elif score >= 50:
        return '#FFB400'  # Orange
    else:
        return '#FF0000'  # Red


def _score_to_color_rgb(score: float) -> Tuple[int, int, int]:
    """Convert score to RGB tuple (for cv2)"""
    hex_color = _score_to_color(score)
    # Convert hex to RGB
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return (r, g, b)


def _score_to_grade(score: float) -> str:
    """Estimate PSA-style grade from edge score"""
    if score >= 95:
        return "PSA 10 (Gem Mint)"
    elif score >= 90:
        return "PSA 9 (Mint)"
    elif score >= 85:
        return "PSA 8 (NM-MT)"
    elif score >= 75:
        return "PSA 7 (NM)"
    elif score >= 65:
        return "PSA 6 (EX-MT)"
    else:
        return "PSA 5 or lower"



import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class SurfaceMetrics:
    """Container for all computed surface features"""
    scratch_density: float
    scratch_severity: float
    wear_uniformity: float
    wear_intensity: float
    defect_count: int
    defect_area_ratio: float
    texture_entropy: float
    gloss_variation: float
    edge_sharpness: float
    
    def to_dict(self) -> Dict[str, float]:
        return self.__dict__


class SurfaceAnalyzer:
    """
    Advanced surface quality analyzer with calibration support.
    Suitable for card grading, manufacturing QC, or surface inspection.
    """
    
    def __init__(self, calibration_data: Optional[Dict] = None):
        """
        Args:
            calibration_data: Reference statistics from pristine samples
                             {'mean': {...}, 'std': {...}}
        """
        self.calibration = calibration_data or self._default_calibration()
    
    def _default_calibration(self) -> Dict:
        """Reasonable defaults - should be replaced with real calibration"""
        return {
            'mean': {
                'scratch_density': 5.0,
                'scratch_severity': 10.0,
                'wear_intensity': 8.0,
                'defect_area_ratio': 0.002
            },
            'std': {
                'scratch_density': 3.0,
                'scratch_severity': 5.0,
                'wear_intensity': 4.0,
                'defect_area_ratio': 0.001
            }
        }
    
    def extract_roi(self, image: np.ndarray, contour: np.ndarray, 
                    margin: float = 0.05) -> Tuple[np.ndarray, bool]:
        """
        Extract region of interest with safety checks.
        
        Returns:
            (roi_image, is_valid)
        """
        x, y, w, h = cv2.boundingRect(contour)
        
        # Validate minimum size
        if w < 50 or h < 50:
            return np.zeros((1, 1, 3), dtype=np.uint8), False
        
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        roi = image[
            max(0, y + margin_y) : min(image.shape[0], y + h - margin_y),
            max(0, x + margin_x) : min(image.shape[1], x + w - margin_x)
        ]
        
        return roi, roi.size > 0
    
    def preprocess(self, roi: np.ndarray) -> np.ndarray:
        """Advanced preprocessing pipeline"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # CLAHE for adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Denoise while preserving edges
        gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        return gray
    
    def detect_scratches(self, gray: np.ndarray) -> Tuple[float, float]:
        """
        Advanced scratch detection using directional filtering and Hough transform.
        
        Returns:
            (density, severity) - number of scratches and their prominence
        """
        # Multi-scale edge detection
        edges_fine = cv2.Canny(gray, 30, 90)
        edges_coarse = cv2.Canny(gray, 50, 150)
        
        # Hough Line Transform for linear features
        lines = cv2.HoughLinesP(
            edges_fine,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=15,
            maxLineGap=3
        )
        
        if lines is None:
            return 0.0, 0.0
        
        # Filter for scratch-like features (high aspect ratio, specific angles)
        scratch_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Scratches are typically longer and more linear
            if length > 20:
                scratch_lines.append(length)
        
        density = len(scratch_lines) / (gray.shape[0] * gray.shape[1] / 10000)
        severity = np.mean(scratch_lines) if scratch_lines else 0.0
        
        return density, severity
    
    def analyze_wear(self, gray: np.ndarray) -> Tuple[float, float]:
        """
        Analyze surface wear using texture analysis.
        
        Returns:
            (uniformity, intensity) - texture consistency and degradation level
        """
        # High-frequency component extraction
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        high_freq = cv2.absdiff(gray, blurred)
        
        # Local standard deviation (texture roughness)
        kernel_size = 15
        local_std = cv2.blur(high_freq.astype(np.float32)**2, (kernel_size, kernel_size))
        local_std = np.sqrt(local_std)
        
        # Uniformity: how consistent is the wear?
        uniformity = 1.0 / (1.0 + np.std(local_std))
        
        # Intensity: overall wear level
        intensity = np.mean(local_std)
        
        return uniformity, intensity
    
    def detect_defects(self, gray: np.ndarray) -> Tuple[int, float]:
        """
        Detect isolated defects (dust, stains, dents).
        
        Returns:
            (count, area_ratio)
        """
        # Adaptive thresholding to handle lighting variations
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        # Morphological operations to isolate defects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Filter defects by size (ignore noise and large shadows)
        valid_defects = 0
        total_defect_area = 0
        
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if 5 < area < 500:  # Reasonable defect size range
                valid_defects += 1
                total_defect_area += area
        
        total_area = gray.shape[0] * gray.shape[1]
        area_ratio = total_defect_area / total_area if total_area > 0 else 0.0
        
        return valid_defects, area_ratio
    
    def calculate_texture_entropy(self, gray: np.ndarray) -> float:
        """
        Calculate texture entropy - higher means more irregular surface.
        """
        hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    def calculate_gloss_variation(self, gray: np.ndarray) -> float:
        """
        Measure gloss variation using gradient analysis.
        """
        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Coefficient of variation
        mean_grad = np.mean(gradient_magnitude)
        std_grad = np.std(gradient_magnitude)
        
        cv_coeff = std_grad / (mean_grad + 1e-6)
        return cv_coeff
    
    def calculate_edge_sharpness(self, gray: np.ndarray) -> float:
        """
        Measure edge sharpness - degraded surfaces have softer edges.
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        return sharpness
    
    def analyze(self, image: np.ndarray, contour: np.ndarray) -> Optional[SurfaceMetrics]:
        """
        Complete surface analysis pipeline.
        
        Returns:
            SurfaceMetrics object or None if analysis fails
        """
        # Extract ROI
        roi, is_valid = self.extract_roi(image, contour)
        if not is_valid:
            return None
        
        # Preprocess
        gray = self.preprocess(roi)
        
        # Extract all features
        scratch_density, scratch_severity = self.detect_scratches(gray)
        wear_uniformity, wear_intensity = self.analyze_wear(gray)
        defect_count, defect_area_ratio = self.detect_defects(gray)
        texture_entropy = self.calculate_texture_entropy(gray)
        gloss_variation = self.calculate_gloss_variation(gray)
        edge_sharpness = self.calculate_edge_sharpness(gray)
        
        return SurfaceMetrics(
            scratch_density=scratch_density,
            scratch_severity=scratch_severity,
            wear_uniformity=wear_uniformity,
            wear_intensity=wear_intensity,
            defect_count=defect_count,
            defect_area_ratio=defect_area_ratio,
            texture_entropy=texture_entropy,
            gloss_variation=gloss_variation,
            edge_sharpness=edge_sharpness
        )
    
    def compute_quality_score(self, metrics: SurfaceMetrics) -> Dict[str, float]:
        """
        Convert metrics to normalized quality scores (0-100).
        
        Returns:
            Dictionary with component scores and final grade
        """
        # Normalize against calibration data
        scratch_score = 100 * np.exp(-0.5 * (
            metrics.scratch_density / self.calibration['mean']['scratch_density']
        ))
        
        wear_score = 100 * np.exp(-0.3 * (
            metrics.wear_intensity / self.calibration['mean']['wear_intensity']
        ))
        
        defect_score = 100 * (1 - np.tanh(
            10 * metrics.defect_area_ratio / self.calibration['mean']['defect_area_ratio']
        ))
        
        # Texture quality (normalized entropy, inverted)
        texture_score = 100 * (1 - np.tanh(metrics.texture_entropy / 8.0))
        
        # Edge preservation score
        edge_score = 100 * np.tanh(metrics.edge_sharpness / 1000.0)
        
        # Weighted final score
        final_score = (
            0.30 * scratch_score +
            0.25 * wear_score +
            0.20 * defect_score +
            0.15 * texture_score +
            0.10 * edge_score
        )
        
        return {
            'scratch_score': round(scratch_score, 2),
            'wear_score': round(wear_score, 2),
            'defect_score': round(defect_score, 2),
            'texture_score': round(texture_score, 2),
            'edge_score': round(edge_score, 2),
            'final_score': round(final_score, 2)
        }
    
    def grade(self, image: np.ndarray, contour: np.ndarray) -> Optional[Dict]:
        """
        Complete grading pipeline.
        
        Returns:
            Dictionary with metrics and scores, or None if analysis fails
        """
        metrics = self.analyze(image, contour)
        if metrics is None:
            return None
        
        scores = self.compute_quality_score(metrics)
        
        return {
            'metrics': metrics.to_dict(),
            'scores': scores
        }
    
    def visualize_analysis(self, image: np.ndarray, contour: np.ndarray, 
                          result: Dict, save_path: Optional[str] = None,
                          dpi: int = 150):
        """
        Create comprehensive visualization of surface analysis results.
        
        Args:
            image: Original image
            contour: Surface contour
            result: Result from grade() method
            save_path: Optional path to save figure
            dpi: Resolution for saved figure
        """
        if result is None:
            print("No valid result to visualize")
            return
        
        # Extract ROI for detailed views
        roi, is_valid = self.extract_roi(image, contour)
        if not is_valid:
            print("Invalid ROI")
            return
        
        gray = self.preprocess(roi)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
        
        # ============================================================
        # Row 1: Original image and ROI
        # ============================================================
        
        # Original image with contour overlay
        ax1 = fig.add_subplot(gs[0, 0:2])
        img_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax1.imshow(img_display)
        ax1.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2, label='Contour')
        
        # Draw bounding box
        x, y, w, h = cv2.boundingRect(contour)
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', 
                        facecolor='none', linestyle='--', label='ROI')
        ax1.add_patch(rect)
        
        ax1.set_title('Original Image with Detection', fontsize=12, fontweight='bold')
        ax1.axis('off')
        ax1.legend(loc='upper right')
        
        # Extracted ROI
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax2.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        ax2.set_title('Extracted ROI (Preprocessed)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # ============================================================
        # Row 2: Feature Detection Visualizations
        # ============================================================
        
        # Scratch detection
        ax3 = fig.add_subplot(gs[1, 0])
        edges = cv2.Canny(gray, 30, 90)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=15, maxLineGap=3)
        
        scratch_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(scratch_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        ax3.imshow(scratch_vis)
        ax3.set_title(f'Scratches Detected\nDensity: {result["metrics"]["scratch_density"]:.2f}', 
                     fontsize=10, fontweight='bold')
        ax3.axis('off')
        
        # Wear analysis (high-frequency component)
        ax4 = fig.add_subplot(gs[1, 1])
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        high_freq = cv2.absdiff(gray, blurred)
        ax4.imshow(high_freq, cmap='hot')
        ax4.set_title(f'Wear Pattern\nIntensity: {result["metrics"]["wear_intensity"]:.2f}', 
                     fontsize=10, fontweight='bold')
        ax4.axis('off')
        
        # Defect detection
        ax5 = fig.add_subplot(gs[1, 2])
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        ax5.imshow(binary, cmap='gray')
        ax5.set_title(f'Defect Map\nCount: {result["metrics"]["defect_count"]}', 
                     fontsize=10, fontweight='bold')
        ax5.axis('off')
        
        # Edge sharpness visualization
        ax6 = fig.add_subplot(gs[1, 3])
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        ax6.imshow(np.abs(laplacian), cmap='viridis')
        ax6.set_title(f'Edge Sharpness\nScore: {result["metrics"]["edge_sharpness"]:.1f}', 
                     fontsize=10, fontweight='bold')
        ax6.axis('off')
        
        # ============================================================
        # Row 3: Scores and Metrics
        # ============================================================
        
        # Score breakdown (bar chart)
        ax7 = fig.add_subplot(gs[2, 0:2])
        scores = result['scores']
        score_names = ['Scratch', 'Wear', 'Defect', 'Texture', 'Edge']
        score_values = [
            scores['scratch_score'],
            scores['wear_score'],
            scores['defect_score'],
            scores['texture_score'],
            scores['edge_score']
        ]
        
        colors = ['#2ecc71' if s >= 80 else '#f39c12' if s >= 60 else '#e74c3c' 
                 for s in score_values]
        
        bars = ax7.barh(score_names, score_values, color=colors, edgecolor='black', linewidth=1.5)
        ax7.set_xlim(0, 100)
        ax7.set_xlabel('Score', fontsize=11, fontweight='bold')
        ax7.set_title('Component Scores Breakdown', fontsize=12, fontweight='bold')
        ax7.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, score_values)):
            ax7.text(val + 2, i, f'{val:.1f}', va='center', fontsize=10, fontweight='bold')
        
        # Final score gauge
        ax8 = fig.add_subplot(gs[2, 2:4])
        final_score = scores['final_score']
        
        # Determine grade
        if final_score >= 90:
            grade = 'MINT'
            grade_color = '#2ecc71'
        elif final_score >= 80:
            grade = 'EXCELLENT'
            grade_color = '#27ae60'
        elif final_score >= 70:
            grade = 'GOOD'
            grade_color = '#f39c12'
        elif final_score >= 60:
            grade = 'FAIR'
            grade_color = '#e67e22'
        else:
            grade = 'POOR'
            grade_color = '#e74c3c'
        
        # Draw gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        ax8.plot(x, y, 'k-', linewidth=3)
        ax8.fill_between(x, 0, y, where=(x <= np.cos(np.pi * (1 - final_score/100))), 
                        color=grade_color, alpha=0.7)
        
        # Needle
        angle = np.pi * (1 - final_score/100)
        ax8.plot([0, np.cos(angle) * 0.9], [0, np.sin(angle) * 0.9], 
                'r-', linewidth=4, marker='o', markersize=10)
        
        # Score text
        ax8.text(0, -0.3, f'{final_score:.1f}', ha='center', va='center', 
                fontsize=36, fontweight='bold', color=grade_color)
        ax8.text(0, -0.5, grade, ha='center', va='center', 
                fontsize=24, fontweight='bold', color=grade_color)
        
        ax8.set_xlim(-1.2, 1.2)
        ax8.set_ylim(-0.7, 1.2)
        ax8.set_aspect('equal')
        ax8.axis('off')
        ax8.set_title('Overall Quality Score', fontsize=12, fontweight='bold')
        
        # Add metrics table
        metrics_text = (
            f"Detailed Metrics:\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Texture Entropy: {result['metrics']['texture_entropy']:.2f}\n"
            f"Gloss Variation: {result['metrics']['gloss_variation']:.3f}\n"
            f"Defect Area: {result['metrics']['defect_area_ratio']*100:.3f}%\n"
            f"Wear Uniformity: {result['metrics']['wear_uniformity']:.3f}"
        )
        
        ax8.text(0, -0.85, metrics_text, ha='center', va='top', 
                fontsize=9, family='monospace', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Overall title
        fig.suptitle('Surface Quality Analysis Report', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import tempfile
import os

# Import all the analysis modules from your code
# (Note: In production, these would be in separate files)

# Set page config
st.set_page_config(
    page_title="Trading Card Grader",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .score-excellent {
        color: #2ecc71;
        font-weight: bold;
    }
    .score-good {
        color: #f39c12;
        font-weight: bold;
    }
    .score-poor {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def load_image(uploaded_file):
    """Load and convert uploaded image to OpenCV format"""
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    return image

def get_score_color(score):
    """Return color based on score"""
    if score >= 90:
        return "score-excellent"
    elif score >= 70:
        return "score-good"
    else:
        return "score-poor"

def get_grade_text(score):
    """Convert score to grade text"""
    if score >= 95:
        return "Gem Mint (10)"
    elif score >= 90:
        return "Mint (9)"
    elif score >= 85:
        return "Near Mint-Mint (8.5)"
    elif score >= 80:
        return "Near Mint (8)"
    elif score >= 75:
        return "Excellent-Mint (7)"
    elif score >= 70:
        return "Excellent (6)"
    elif score >= 60:
        return "Good (5)"
    elif score >= 50:
        return "Fair (4)"
    else:
        return "Poor (1-3)"

# Main app
def main():
    st.markdown('<h1 class="main-header">🎴 Trading Card Grading System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This advanced AI-powered system analyzes trading cards across multiple quality dimensions:
    - **Centering**: Border alignment and symmetry
    - **Corners**: Wear, whitening, and damage detection
    - **Edges**: Surface quality and wear patterns
    - **Surface**: Scratches, defects, and overall condition
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        
        st.subheader("Modules to Run")
        run_centering = st.checkbox("Centering Analysis", value=True)
        run_corners = st.checkbox("Corner Analysis", value=True)
        run_edges = st.checkbox("Edge Analysis", value=True)
        run_surface = st.checkbox("Surface Analysis", value=True)
        
        st.divider()
        
        st.subheader("Visualization Options")
        show_visualizations =True
        show_metrics = False
        
        st.divider()
        
        st.subheader("About")
        st.info("""
        This system uses advanced computer vision techniques including:
        - Contour detection
        - Gradient analysis
        - Multi-scale feature extraction
        - Statistical modeling
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a trading card image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear, well-lit image of your trading card"
    )
    
    if uploaded_file is not None:
        # Load image
            image = load_image(uploaded_file)
            
            # Display original image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Original Image")
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(display_image, width='stretch')
            
            # Detect card contour
            with st.spinner("🔍 Detecting card boundaries..."):
                try:

                    detector = AdvancedCardDetector()
                    contour, debug_info = detector.detect_card_contour(image, debug=True)
                    
                    # Draw contour on image
                    contour_image = image.copy()
                    cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 3)
                    
                    with col2:
                        st.subheader("✅ Card Detected")
                        display_contour = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
                        st.image(display_contour, width='stretch')
                        
                        if debug_info:
                            with st.expander("Detection Details"):
                                st.write(f"**Confidence**: {debug_info['confidence']:.2%}")
                                st.write(f"**Area**: {debug_info['area']:.0f} pixels")
                                st.write(f"**Aspect Ratio**: {debug_info['aspect_ratio']:.2f}")
                    
                except Exception as e:
                    st.error(f"❌ Could not detect card: {str(e)}")
                    st.info("Please ensure the card is clearly visible and takes up most of the image.")
                    return
            
            st.divider()
            
            # Analysis sections
            results = {}
            
            # Centering Analysis
            # Centering Analysis
            if run_centering:
                    st.header("🎯 Centering Analysis")

                    with st.spinner("Analyzing centering..."):
                                        centering_score, centering_details = analyze_card_centering(image, contour)
                                        results["centering"] = centering_score
                                        col1, col2, col3 = st.columns(3)
                                        print("centering_score: ",centering_score)
                                        st.info("Overall Score")
                    
                                        with col1:
                                           st.metric(
                                           "Overall Score",
                                           f"{centering_score:.1f}/100"
                                           )

                                        with col2:
                                                            st.metric(
                                                            "L/R Ratio",
                                                            f"{centering_details.get('lr_ratio', 0):.1f}%"
                                                            )

                                        with col3:
                                                            st.metric(
                                                            "T/B Ratio",
                                                            f"{centering_details.get('tb_ratio', 0):.1f}%"
                                                            )
                                        if show_metrics:
                                                            with st.expander("Detailed Centering Metrics"):
                                                                  st.json(centering_details)


  
            # Corner Analysis
            if run_corners:
                st.header("📐 Corner Analysis")
                
                with st.spinner("Analyzing corners..."):
                    try:
                        corner_score, corner_details = analyze_corners_advanced(image, contour)
                        results['corners'] = corner_score
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Overall Score",
                                f"{corner_score:.1f}/100"
                            )
                        
                    

                        # Individual corner scores
                        st.subheader("Individual Corner Scores")
                        cols = st.columns(4)
                        
                        for idx, corner_data in enumerate(corner_details['corners']):
                            with cols[idx]:
                                score = corner_data['score']
                                st.markdown(f"**{corner_data['name']}**")
                                st.markdown(f"<p class='{get_score_color(score)}'>{score:.1f}/100</p>", 
                                          unsafe_allow_html=True)
                        
                        if show_visualizations:
                            st.subheader("Corner Visualization")
            
                        if show_metrics:
                            with st.expander("Detailed Corner Metrics"):
                                st.json(corner_details)
                        
                    except Exception as e:
                        st.error(f"Corner analysis failed: {str(e)}")
            
            # Edge Analysis
            if run_edges:
                st.header("📏 Edge Analysis")
                with st.spinner("Analyzing edges..."):
                        edge_analyzer = AdvancedEdgeAnalyzer()
                        edge_score, edge_details = edge_analyzer.analyze_edges(image, contour)
                        print('==================')
                        print( edge_score, edge_details)
                        results['edges'] = edge_score
                        
                        col1, col2, col3, col4 = st.columns(4)
                        print("edge_score: ",edge_score)
                        st.info("Overall Score")
                        st.info(f"{edge_score:.1f}/100")
                        with col1:
                                        st.metric(
                                        "Overall Score",
                                        f"{edge_score:.1f}/100"
                                        )
            # Surface Analysis
            if run_surface:
                st.header("🔬 Surface Analysis")
                
                with st.spinner("Analyzing surface quality..."):
                        surface_analyzer = SurfaceAnalyzer()
                        surface_result = surface_analyzer.grade(image, contour)
                        
                        if surface_result:
                            surface_score = surface_result['scores']['final_score']
                            results['surface'] = surface_score
                            
                            col21, col22, col23, col24 = st.columns(4)
                            
                            with col21:
                                st.metric(
                                    "Overall Score",
                                    f"{surface_score:.1f}/100"
                             )
                            
                    

    
    else:
        st.info("👆 Please upload a trading card image to begin analysis")

if __name__ == "__main__":
    main()



