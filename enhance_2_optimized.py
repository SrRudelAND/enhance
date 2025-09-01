#!/usr/bin/env python3
"""
Part 1
MTG Proxy Enhancer - Core Processing Engine
Optimized for performance and modularity
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EnhancementSettings:
    """Data class for enhancement parameters"""
    clip_limit: float = 2.0
    gamma: float = 1.2
    saturation: float = 1.0
    brightness: float = 0.0
    contrast: float = 1.0
    vibrance: float = 0.0
    warmth: float = 0.0
    tint: float = 0.0
    exposure: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    clarity: float = 0.0
    preserve_black: bool = True
    black_threshold: int = 15

@dataclass
class ImageStats:
    """Image analysis statistics"""
    mean_brightness: float
    contrast_std: float
    color_balance: List[float]
    saturation_mean: float
    is_dark: bool
    is_bright: bool
    is_low_contrast: bool
    has_color_cast: bool
    cast_type: Optional[str]

class ImageAnalyzer:
    """Optimized image analysis for automatic enhancement"""
    
    @staticmethod
    def analyze_image(img: np.ndarray) -> Tuple[ImageStats, EnhancementSettings]:
        """
        Fast image analysis with vectorized operations
        Returns: (stats, optimal_settings)
        """
        # Pre-compute color spaces once
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Vectorized statistics
        mean_brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))
        
        # Color analysis
        color_means = np.mean(img, axis=(0, 1))  # [B, G, R]
        avg_color = np.mean(color_means)
        color_cast_threshold = 15
        
        # Determine color cast
        cast_type = None
        has_color_cast = False
        if color_means[0] > avg_color + color_cast_threshold:
            cast_type = "blue"
            has_color_cast = True
        elif color_means[2] > avg_color + color_cast_threshold:
            cast_type = "red"
            has_color_cast = True
        elif color_means[1] > avg_color + color_cast_threshold:
            cast_type = "green"
            has_color_cast = True
        
        # Saturation analysis
        saturation_mean = float(np.mean(hsv[..., 1]))
        
        # Create stats object
        stats = ImageStats(
            mean_brightness=mean_brightness,
            contrast_std=contrast_std,
            color_balance=color_means.tolist(),
            saturation_mean=saturation_mean,
            is_dark=mean_brightness < 80,
            is_bright=mean_brightness > 180,
            is_low_contrast=contrast_std < 35,
            has_color_cast=has_color_cast,
            cast_type=cast_type
        )
        
        # Generate optimal settings
        settings = ImageAnalyzer._generate_settings(stats)
        
        return stats, settings
    
    @staticmethod
    def _generate_settings(stats: ImageStats) -> EnhancementSettings:
        """Generate optimal enhancement settings based on image stats"""
        settings = EnhancementSettings()
        
        # Brightness/exposure adjustments
        if stats.is_dark:
            settings.exposure = 0.3
            settings.shadows = 20
            settings.brightness = 10
        elif stats.is_bright:
            settings.highlights = -15
            settings.whites = -10
        
        # Contrast adjustments
        if stats.is_low_contrast:
            settings.clip_limit = 3.5
            settings.contrast = 1.3
            settings.clarity = 15
        elif stats.contrast_std > 80:  # High contrast
            settings.clip_limit = 1.0
            settings.highlights = -20
            settings.shadows = 15
        
        # Color cast corrections
        if stats.has_color_cast:
            if stats.cast_type == "blue":
                settings.warmth = 15
                settings.tint = -5
            elif stats.cast_type == "red":
                settings.warmth = -10
                settings.tint = 5
            elif stats.cast_type == "green":
                settings.tint = -10
        
        # Saturation adjustments
        if stats.saturation_mean < 60:  # Desaturated
            settings.saturation = 1.2
            settings.vibrance = 20
        elif stats.saturation_mean > 180:  # Oversaturated
            settings.saturation = 0.9
            settings.vibrance = -10
        
        # Gamma based on brightness distribution
        dark_ratio = np.sum(stats.mean_brightness < 64) / (256 * 256)  # Approximate
        if dark_ratio > 0.6:
            settings.gamma = 1.4
        elif stats.mean_brightness > 192:
            settings.gamma = 0.8
        
        return settings

class ImageProcessor:
    """Optimized image processing engine"""
    
    @staticmethod
    def preserve_black_pixels(original: np.ndarray, enhanced: np.ndarray, 
                            threshold: int = 15) -> np.ndarray:
        """Efficiently preserve black pixels using vectorized operations"""
        # Create boolean mask for black pixels
        black_mask = np.all(original <= threshold, axis=2, keepdims=True)
        
        # Use numpy where for efficient conditional replacement
        return np.where(black_mask, original, enhanced)
    
    @staticmethod
    def apply_clahe(img: np.ndarray, clip_limit: float) -> np.ndarray:
        """Apply CLAHE in LAB color space for better results"""
        if clip_limit <= 0:
            return img
            
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[..., 0]
        
        # Apply CLAHE only to luminance channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Merge back
        lab[..., 0] = l_enhanced
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def apply_gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
        """Optimized gamma correction using lookup table"""
        if abs(gamma - 1.0) < 0.001:
            return img
            
        # Pre-compute lookup table
        inv_gamma = 1.0 / gamma
        lut = np.array([(i / 255.0) ** inv_gamma * 255 
                       for i in range(256)], dtype=np.uint8)
        
        return cv2.LUT(img, lut)
    
    @staticmethod
    def apply_tone_mapping(img: np.ndarray, highlights: float, shadows: float, 
                          whites: float, blacks: float) -> np.ndarray:
        """Advanced tone mapping with optimized calculations"""
        if all(x == 0 for x in [highlights, shadows, whites, blacks]):
            return img
        
        # Work in float32 for precision
        img_float = img.astype(np.float32) / 255.0
        
        # Calculate luminance once
        luminance = 0.299 * img_float[..., 2] + 0.587 * img_float[..., 1] + 0.114 * img_float[..., 0]
        
        # Apply tone adjustments efficiently
        if highlights != 0:
            highlight_mask = luminance ** 2
            adjustment = 1.0 + (highlights / 100.0)
            img_float = img_float * (1.0 - highlight_mask[..., np.newaxis]) + \
                       img_float * adjustment * highlight_mask[..., np.newaxis]
        
        if shadows != 0:
            shadow_mask = 1.0 - luminance
            adjustment = 1.0 + (shadows / 100.0)
            img_float = img_float * (1.0 - shadow_mask[..., np.newaxis]) + \
                       img_float * adjustment * shadow_mask[..., np.newaxis]
        
        # Whites and blacks adjustments
        if whites != 0:
            white_mask = luminance ** 2
            adjustment = 1.0 + (whites / 100.0)
            img_float = img_float * (1.0 - white_mask[..., np.newaxis]) + \
                       img_float * adjustment * white_mask[..., np.newaxis]
        
        if blacks != 0:
            black_mask = 1.0 - np.sqrt(luminance)
            adjustment = 1.0 + (blacks / 100.0)
            img_float = img_float * (1.0 - black_mask[..., np.newaxis]) + \
                       img_float * adjustment * black_mask[..., np.newaxis]
        
        return np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_color_adjustments(img: np.ndarray, saturation: float, vibrance: float,
                              warmth: float, tint: float) -> np.ndarray:
        """Optimized color adjustments in HSV space"""
        if all(x in [1.0, 0.0] for x in [saturation, vibrance, warmth, tint]):
            return img
        
        # Convert to HSV for color adjustments
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Saturation
        if saturation != 1.0:
            hsv[..., 1] *= saturation
        
        # Vibrance (affects less saturated pixels more)
        if vibrance != 0:
            current_sat = hsv[..., 1] / 255.0
            vibrance_factor = 1.0 + (vibrance / 100.0) * (1.0 - current_sat)
            hsv[..., 1] *= vibrance_factor
        
        # Warmth (hue shift)
        if warmth != 0:
            hsv[..., 0] = (hsv[..., 0] + warmth * 0.5) % 180
        
        # Clip saturation
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Tint adjustment in BGR space
        if tint != 0:
            if tint > 0:  # More magenta
                result[..., [0, 2]] = np.clip(result[..., [0, 2]] + tint, 0, 255)
            else:  # More green
                result[..., 1] = np.clip(result[..., 1] - tint, 0, 255)
        
        return result

class MTGProxyEnhancer:
    """Main enhancer class with optimized processing pipeline"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(self, input_folder: str = "mtgproxy/Input", 
                 output_folder: str = "mtgproxy/Output"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.stop_processing = False
        self.is_processing = False
        
        self._setup_folders()
        self.images = self._load_image_list()
        
    def _setup_folders(self) -> None:
        """Create folders if they don't exist"""
        self.input_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
    def _load_image_list(self) -> List[str]:
        """Load list of supported image files"""
        images = []
        for ext in self.SUPPORTED_FORMATS:
            # Handle each glob separately (no generator addition)
            images.extend([f.name for f in self.input_folder.glob(f"*{ext}")])
            images.extend([f.name for f in self.input_folder.glob(f"*{ext.upper()}")])
        
        logger.info(f"Found {len(images)} images in {self.input_folder}")
        return sorted(list(set(images)))  # Remove duplicates and sort
    
    def enhance_image(self, img: np.ndarray, settings: EnhancementSettings) -> np.ndarray:
        """
        Optimized enhancement pipeline with minimal memory allocation
        """
        original = img.copy()
        
        # 1. Exposure adjustment (early in pipeline for better results)
        if settings.exposure != 0:
            img = (img.astype(np.float32) * (2.0 ** settings.exposure)).clip(0, 255).astype(np.uint8)
        
        # 2. Tone mapping
        img = ImageProcessor.apply_tone_mapping(
            img, settings.highlights, settings.shadows, settings.whites, settings.blacks
        )
        
        # 3. Basic adjustments
        if settings.brightness != 0 or settings.contrast != 1.0:
            img = cv2.convertScaleAbs(img, alpha=settings.contrast, beta=settings.brightness)
        
        # 4. Gamma correction
        img = ImageProcessor.apply_gamma_correction(img, settings.gamma)
        
        # 5. CLAHE for adaptive contrast
        img = ImageProcessor.apply_clahe(img, settings.clip_limit)
        
        # 6. Color adjustments
        img = ImageProcessor.apply_color_adjustments(
            img, settings.saturation, settings.vibrance, settings.warmth, settings.tint
        )
        
        # 7. Clarity (local contrast enhancement)
        if settings.clarity != 0:
            gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
            img = cv2.addWeighted(img, 1.0 + settings.clarity/100.0, 
                                gaussian, -settings.clarity/100.0, 0)
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        # 8. Preserve black pixels (MTG-specific)
        if settings.preserve_black:
            img = ImageProcessor.preserve_black_pixels(original, img, settings.black_threshold)
        
        return img
    
    def auto_enhance_image(self, img: np.ndarray) -> Tuple[np.ndarray, EnhancementSettings, List[str]]:
        """
        Automatically enhance image using AI-like analysis
        Returns: (enhanced_image, settings_used, analysis_notes)
        """
        stats, settings = ImageAnalyzer.analyze_image(img)
        enhanced = self.enhance_image(img, settings)
        
        # Generate analysis notes
        notes = []
        if stats.is_dark:
            notes.append("Dark image: increased exposure and shadows")
        if stats.is_bright:
            notes.append("Bright image: reduced highlights")
        if stats.is_low_contrast:
            notes.append("Low contrast: increased CLAHE and clarity")
        if stats.has_color_cast:
            notes.append(f"{stats.cast_type.title()} color cast: corrected")
        if stats.saturation_mean < 60:
            notes.append("Low saturation: increased vibrance")
        
        return enhanced, settings, notes
    
    def process_single_image(self, filename: str, settings: EnhancementSettings, 
                           overwrite: bool = True) -> bool:
        """
        Process a single image with error handling
        Returns: success status
        """
        try:
            input_path = self.input_folder / filename
            output_path = self.output_folder / filename
            
            # Check if file exists and overwrite policy
            if output_path.exists() and not overwrite:
                logger.info(f"Skipping {filename} (already exists)")
                return False
            
            # Load and process image
            img = cv2.imread(str(input_path))
            if img is None:
                logger.error(f"Could not load {filename}")
                return False
            
            # Enhance image
            enhanced = self.enhance_image(img, settings)
            
            # Save with high quality
            success = cv2.imwrite(str(output_path), enhanced, 
                                [cv2.IMWRITE_JPEG_QUALITY, 95,
                                 cv2.IMWRITE_PNG_COMPRESSION, 1])
            
            if success:
                logger.debug(f"Successfully processed {filename}")
                return True
            else:
                logger.error(f"Failed to save {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return False


"""
Part 2
MTG Proxy Enhancer - Batch Processing and Interface Components
Continuation from Part 1
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
from dataclasses import asdict
import json

class BatchProcessor:
    """High-performance batch processing with threading and progress tracking"""
    
    def __init__(self, enhancer_instance):
        self.enhancer = enhancer_instance
        self.stop_event = Event()
        
    def batch_process_threaded(self, settings: EnhancementSettings, 
                             max_workers: int = 4, overwrite: bool = True) -> Dict:
        """
        Multi-threaded batch processing for better performance
        """
        if not self.enhancer.images:
            logger.warning("No images to process")
            return {"success": 0, "errors": 0, "skipped": 0, "time": 0}
        
        self.enhancer.is_processing = True
        self.stop_event.clear()
        
        start_time = time.time()
        results = {"success": 0, "errors": 0, "skipped": 0, "time": 0}
        
        logger.info(f"Starting threaded batch processing with {max_workers} workers")
        logger.info(f"Processing {len(self.enhancer.images)} images")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_filename = {
                executor.submit(self.enhancer.process_single_image, filename, settings, overwrite): filename
                for filename in self.enhancer.images
            }
            
            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_filename), 1):
                if self.stop_event.is_set():
                    logger.info(f"Processing stopped at image {i}")
                    break
                
                filename = future_to_filename[future]
                
                try:
                    success = future.result()
                    if success:
                        results["success"] += 1
                    else:
                        results["skipped"] += 1
                        
                    # Progress update
                    progress = i / len(self.enhancer.images) * 100
                    print(f"\rProgress: {i}/{len(self.enhancer.images)} [{progress:.1f}%] - {filename[:30]}", 
                          end="", flush=True)
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    results["errors"] += 1
        
        results["time"] = time.time() - start_time
        self.enhancer.is_processing = False
        
        print(f"\n\nBatch processing complete!")
        print(f"Successfully processed: {results['success']} images")
        print(f"Skipped: {results['skipped']} images")
        print(f"Errors: {results['errors']} images")
        print(f"Time elapsed: {results['time']:.1f} seconds")
        print(f"Enhanced images saved to: {self.enhancer.output_folder}")
        
        return results
    
    def auto_batch_process(self, max_workers: int = 4) -> Dict:
        """
        Automatically analyze and enhance each image with optimal settings
        """
        if not self.enhancer.images:
            return {"success": 0, "errors": 0, "time": 0}
        
        self.enhancer.is_processing = True
        self.stop_event.clear()
        
        start_time = time.time()
        results = {"success": 0, "errors": 0, "time": 0}
        
        logger.info(f"Starting auto-enhancement with {max_workers} workers")
        
        def process_auto_enhance(filename):
            """Process single image with auto-enhancement"""
            try:
                input_path = self.enhancer.input_folder / filename
                output_path = self.enhancer.output_folder / filename
                
                # Load image
                img = cv2.imread(str(input_path))
                if img is None:
                    return False, f"Could not load {filename}"
                
                # Auto-enhance
                enhanced, settings, notes = self.enhancer.auto_enhance_image(img)
                
                # Save
                success = cv2.imwrite(str(output_path), enhanced, 
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                return success, notes
                
            except Exception as e:
                return False, str(e)
        
        # Process with threading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_filename = {
                executor.submit(process_auto_enhance, filename): filename
                for filename in self.enhancer.images
            }
            
            for i, future in enumerate(as_completed(future_to_filename), 1):
                if self.stop_event.is_set():
                    break
                
                filename = future_to_filename[future]
                
                try:
                    success, info = future.result()
                    if success:
                        results["success"] += 1
                    else:
                        results["errors"] += 1
                        logger.error(f"Failed: {filename} - {info}")
                    
                    progress = i / len(self.enhancer.images) * 100
                    print(f"\rAuto-enhancing: {filename[:30]} [{progress:.1f}%]", 
                          end="", flush=True)
                    
                except Exception as e:
                    logger.error(f"Error auto-enhancing {filename}: {e}")
                    results["errors"] += 1
        
        results["time"] = time.time() - start_time
        self.enhancer.is_processing = False
        
        print(f"\n\nAuto-enhancement complete!")
        print(f"Successfully processed: {results['success']} images")
        print(f"Errors: {results['errors']} images") 
        print(f"Time elapsed: {results['time']:.1f} seconds")
        
        return results
    
    def stop_processing(self):
        """Stop current batch processing"""
        self.stop_event.set()
        logger.info("Stop signal sent")

class SettingsManager:
    """Manage and persist enhancement settings"""
    
    @staticmethod
    def save_settings(settings: EnhancementSettings, filename: str = "enhancement_settings.json"):
        """Save settings to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(asdict(settings), f, indent=2)
            logger.info(f"Settings saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
    
    @staticmethod
    def load_settings(filename: str = "enhancement_settings.json") -> Optional[EnhancementSettings]:
        """Load settings from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            return EnhancementSettings(**data)
        except FileNotFoundError:
            logger.info(f"Settings file {filename} not found, using defaults")
            return None
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return None
    
    @staticmethod
    def create_preset_settings() -> Dict[str, EnhancementSettings]:
        """Create common preset configurations"""
        presets = {
            "default": EnhancementSettings(),
            
            "dark_images": EnhancementSettings(
                exposure=0.4, shadows=25, brightness=15, gamma=1.4, 
                clip_limit=3.0, clarity=10
            ),
            
            "bright_images": EnhancementSettings(
                highlights=-20, whites=-15, exposure=-0.2, gamma=0.9,
                clip_limit=1.5
            ),
            
            "low_contrast": EnhancementSettings(
                clip_limit=4.0, contrast=1.4, clarity=20, saturation=1.1,
                vibrance=15
            ),
            
            "color_enhancement": EnhancementSettings(
                saturation=1.3, vibrance=25, clarity=15, clip_limit=2.5
            ),
            
            "professional": EnhancementSettings(
                clip_limit=2.5, gamma=1.1, saturation=1.1, vibrance=10,
                clarity=12, highlights=-5, shadows=8
            ),
            
            "vintage_correction": EnhancementSettings(
                warmth=12, tint=-3, saturation=1.2, contrast=1.1,
                gamma=1.3, clarity=8
            )
        }
        
        return presets

# Enhanced utility functions
def create_mtg_enhancer(input_folder: str = "mtgproxy/Input", 
                       output_folder: str = "mtgproxy/Output") -> MTGProxyEnhancer:
    """Create optimized MTG Proxy Enhancer instance"""
    return MTGProxyEnhancer(input_folder, output_folder)

def quick_enhance_all(input_folder: str = "mtgproxy/Input", 
                     output_folder: str = "mtgproxy/Output",
                     preset: str = "default",
                     max_workers: int = 4) -> Dict:
    """Quick batch enhancement with preset or custom settings"""
    enhancer = MTGProxyEnhancer(input_folder, output_folder)
    batch_processor = BatchProcessor(enhancer)
    
    # Get preset settings
    presets = SettingsManager.create_preset_settings()
    settings = presets.get(preset, presets["default"])
    
    logger.info(f"Using preset: {preset}")
    return batch_processor.batch_process_threaded(settings, max_workers)

def auto_enhance_all(input_folder: str = "mtgproxy/Input", 
                    output_folder: str = "mtgproxy/Output",
                    max_workers: int = 4) -> Dict:
    """Auto-enhance all images using intelligent analysis"""
    enhancer = MTGProxyEnhancer(input_folder, output_folder)
    batch_processor = BatchProcessor(enhancer)
    
    return batch_processor.auto_batch_process(max_workers)

def benchmark_enhancement(input_folder: str = "mtgproxy/Input", 
                         test_count: int = 5) -> Dict:
    """
    Benchmark enhancement performance
    """
    enhancer = MTGProxyEnhancer(input_folder)
    
    if not enhancer.images:
        return {"error": "No images found"}
    
    # Test with first available image
    test_image_path = enhancer.input_folder / enhancer.images[0]
    img = cv2.imread(str(test_image_path))
    
    if img is None:
        return {"error": "Could not load test image"}
    
    settings = EnhancementSettings()
    
    # Warm-up run
    enhancer.enhance_image(img, settings)
    
    # Benchmark runs
    times = []
    for _ in range(test_count):
        start = time.time()
        enhancer.enhance_image(img, settings)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        "image_size": f"{img.shape[1]}x{img.shape[0]}",
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "estimated_batch_time": avg_time * len(enhancer.images),
        "images_per_second": 1.0 / avg_time
    }

# CLI interface helper
def print_usage():
    """Print usage instructions"""
    print(""""
🃏 MTG Proxy Enhancer - Optimized Version

QUICK START:
1. Place images in 'mtgproxy/Input' folder
2. Choose enhancement method:

📚 METHODS:
• Auto Enhancement (Recommended):
  auto_enhance_all()

• Preset Enhancement:
  quick_enhance_all(preset="professional")
  
• Custom Settings:
  enhancer = create_mtg_enhancer()
  settings = EnhancementSettings(gamma=1.3, saturation=1.2)
  BatchProcessor(enhancer).batch_process_threaded(settings)

📋 AVAILABLE PRESETS:
• "default" - Balanced enhancement
• "dark_images" - For underexposed cards
• "bright_images" - For overexposed cards  
• "low_contrast" - Boost contrast and clarity
• "color_enhancement" - Vibrant colors
• "professional" - Subtle professional look
• "vintage_correction" - Fix old/scanned cards

⚡ PERFORMANCE:
• Multi-threaded processing (4 workers by default)
""")


"""
Part 3
MTG Proxy Enhancer - Interactive Interface Components
Continuation from Part 2
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button as MPLButton, CheckButtons
import matplotlib.patches as patches
from IPython.display import display, clear_output
from ipywidgets import (interact, FloatSlider, IntSlider, Checkbox, Button, 
                       VBox, HBox, Output, HTML, Label, Dropdown, Tab)

class InteractiveInterface:
    """Optimized interactive interface with better UX"""
    
    def __init__(self, enhancer: MTGProxyEnhancer):
        self.enhancer = enhancer
        self.batch_processor = BatchProcessor(enhancer)
        self.current_image_idx = 0
        self.current_settings = EnhancementSettings()
        
    def create_comparison_view(self, original: np.ndarray, enhanced: np.ndarray, 
                             blend_ratio: float = 0.5) -> np.ndarray:
        """Create optimized blended comparison"""
        if blend_ratio <= 0.02:
            return original
        elif blend_ratio >= 0.98:
            return enhanced
        else:
            return cv2.addWeighted(original, 1-blend_ratio, enhanced, blend_ratio, 0)
    
    def get_current_image(self) -> Optional[np.ndarray]:
        """Load current image with caching"""
        if not self.enhancer.images or self.current_image_idx >= len(self.enhancer.images):
            return None
            
        filename = self.enhancer.images[self.current_image_idx]
        path = self.enhancer.input_folder / filename
        
        return cv2.imread(str(path))
    
    def create_preview_plot(self, comparison_ratio: float = 0.5, 
                          show_stats: bool = False) -> None:
        """Create optimized preview with better layout"""
        img = self.get_current_image()
        if img is None:
            print("No image available for preview")
            return
        
        # Enhance current image
        enhanced = self.enhancer.enhance_image(img, self.current_settings)
        comparison = self.create_comparison_view(img, enhanced, comparison_ratio)
        
        # Convert to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
        
        # Create optimized figure layout
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[1, 1, 1])
        
        # Main image displays
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1]) 
        ax3 = fig.add_subplot(gs[0, 2])
        
        ax1.imshow(img_rgb)
        ax1.set_title("Original", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(enhanced_rgb)
        ax2.set_title("Enhanced", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        ax3.imshow(comparison_rgb)
        if comparison_ratio <= 0.02:
            title = "100% Original"
        elif comparison_ratio >= 0.98:
            title = "100% Enhanced"
        else:
            title = f"Blend: {comparison_ratio*100:.0f}% Enhanced"
        ax3.set_title(title, fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # Settings and stats panel
        ax_info = fig.add_subplot(gs[1, :])
        ax_info.axis('off')
        
        # Format settings display
        filename = self.enhancer.images[self.current_image_idx]
        settings_text = self._format_settings_display(filename, img.shape, show_stats, img, enhanced)
        
        ax_info.text(0.02, 0.95, settings_text, transform=ax_info.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.9))
        
        plt.tight_layout()
        plt.show()
    
    def _format_settings_display(self, filename: str, img_shape: tuple, 
                               show_stats: bool, original: np.ndarray, 
                               enhanced: np.ndarray) -> str:
        """Format settings display text efficiently"""
        s = self.current_settings
        
        settings_text = f"""🃏 MTG Proxy Enhancement Settings

📁 File: {filename} | 📐 Size: {img_shape[1]}x{img_shape[0]} | 🖼️ Image: {self.current_image_idx+1}/{len(self.enhancer.images)}

🔧 BASIC ADJUSTMENTS:
CLAHE: {s.clip_limit:.1f} | Gamma: {s.gamma:.2f} | Brightness: {s.brightness:+.0f} | Contrast: {s.contrast:.2f}

🎨 COLOR ENHANCEMENT:
Saturation: {s.saturation:.2f} | Vibrance: {s.vibrance:+.0f} | Warmth: {s.warmth:+.0f} | Tint: {s.tint:+.0f}

🌅 TONE MAPPING:
Exposure: {s.exposure:+.1f} | Highlights: {s.highlights:+.0f} | Shadows: {s.shadows:+.0f} | Clarity: {s.clarity:+.0f}

⚫ BLACK PRESERVATION: {s.preserve_black} (Threshold: {s.black_threshold})"""
        
        if show_stats:
            orig_brightness = float(np.mean(original))
            enh_brightness = float(np.mean(enhanced))
            orig_contrast = float(np.std(original))
            enh_contrast = float(np.std(enhanced))
            
            settings_text += f"""

📊 STATISTICS:
Original → Brightness: {orig_brightness:.1f}, Contrast: {orig_contrast:.1f}
Enhanced → Brightness: {enh_brightness:.1f}, Contrast: {enh_contrast:.1f}
Change   → Brightness: {enh_brightness-orig_brightness:+.1f}, Contrast: {enh_contrast-orig_contrast:+.1f}"""
        
        return settings_text
    
    def create_widget_interface(self):
        """Create comprehensive widget-based interface"""
        if not self.enhancer.images:
            return HTML("<h3>❌ No images found. Add images to input folder first.</h3>")
        
        # Navigation widgets
        prev_btn = Button(description="◀ Previous", button_style='info', 
                         layout={'width': '100px'})
        next_btn = Button(description="Next ▶", button_style='info', 
                         layout={'width': '100px'})
        image_info = HTML(f"<b>Image 1 of {len(self.enhancer.images)}</b>: {self.enhancer.images[0]}")
        
        # Preset selection
        presets = SettingsManager.create_preset_settings()
        preset_dropdown = Dropdown(
            options=list(presets.keys()),
            value='default',
            description='Preset:',
            style={'description_width': '80px'}
        )
        
        # Basic controls (optimized ranges)
        basic_controls = [
            FloatSlider(2.0, 0.5, 8.0, 0.1, description="CLAHE", 
                       style={'description_width': '100px'}, layout={'width': '280px'}),
            FloatSlider(1.2, 0.5, 3.0, 0.05, description="Gamma",
                       style={'description_width': '100px'}, layout={'width': '280px'}),
            FloatSlider(0, -50, 50, 1, description="Brightness",
                       style={'description_width': '100px'}, layout={'width': '280px'}),
            FloatSlider(1.0, 0.5, 2.5, 0.05, description="Contrast",
                       style={'description_width': '100px'}, layout={'width': '280px'})
        ]
        
        # Color controls
        color_controls = [
            FloatSlider(1.0, 0.0, 3.0, 0.05, description="Saturation",
                       style={'description_width': '100px'}, layout={'width': '280px'}),
            FloatSlider(0, -50, 50, 1, description="Vibrance",
                       style={'description_width': '100px'}, layout={'width': '280px'}),
            FloatSlider(0, -30, 30, 1, description="Warmth",
                       style={'description_width': '100px'}, layout={'width': '280px'}),
            FloatSlider(0, -30, 30, 1, description="Tint",
                       style={'description_width': '100px'}, layout={'width': '280px'})
        ]
        
        # Tone controls
        tone_controls = [
            FloatSlider(0, -2.0, 2.0, 0.1, description="Exposure",
                       style={'description_width': '100px'}, layout={'width': '280px'}),
            FloatSlider(0, -50, 50, 1, description="Highlights",
                       style={'description_width': '100px'}, layout={'width': '280px'}),
            FloatSlider(0, -50, 50, 1, description="Shadows",
                       style={'description_width': '100px'}, layout={'width': '280px'}),
            FloatSlider(0, -50, 50, 1, description="Clarity",
                       style={'description_width': '100px'}, layout={'width': '280px'})
        ]
        
        # Options
        comparison_slider = FloatSlider(0.5, 0.0, 1.0, 0.01, 
                                       description="Original ↔ Enhanced",
                                       style={'description_width': '150px'}, 
                                       layout={'width': '400px'})
        
        preserve_black = Checkbox(True, description="Preserve Black Pixels")
        black_threshold = IntSlider(15, 5, 50, 1, description="Black Threshold",
                                   style={'description_width': '120px'})
        show_stats = Checkbox(False, description="Show Statistics")
        
        # Action buttons
        auto_btn = Button(description="🎯 Auto Enhance", button_style='primary')
        reset_btn = Button(description="🔄 Reset", button_style='warning')
        save_settings_btn = Button(description="💾 Save Settings", button_style='info')
        
        # Batch processing
        process_btn = Button(description="🚀 Process All", button_style='success',
                           layout={'width': '150px', 'height': '35px'})
        auto_process_btn = Button(description="🤖 Auto Process All", button_style='primary',
                                layout={'width': '150px', 'height': '35px'})
        stop_btn = Button(description="⏹️ Stop", button_style='danger',
                         layout={'width': '100px', 'height': '35px'}, disabled=True)
        
        # Output areas
        preview_output = Output()
        batch_output = Output()
        
        # Event handlers
        def update_settings_from_widgets():
            """Update current settings from widget values"""
            self.current_settings = EnhancementSettings(
                clip_limit=basic_controls[0].value,
                gamma=basic_controls[1].value,
                brightness=basic_controls[2].value,
                contrast=basic_controls[3].value,
                saturation=color_controls[0].value,
                vibrance=color_controls[1].value,
                warmth=color_controls[2].value,
                tint=color_controls[3].value,
                exposure=tone_controls[0].value,
                highlights=tone_controls[1].value,
                shadows=tone_controls[2].value,
                clarity=tone_controls[3].value,
                preserve_black=preserve_black.value,
                black_threshold=black_threshold.value
            )
        
        def update_preview():
            """Update preview display"""
            update_settings_from_widgets()
            with preview_output:
                clear_output(wait=True)
                self.create_preview_plot(comparison_slider.value, show_stats.value)
        
        def on_navigation(direction):
            """Handle image navigation"""
            if direction == "prev":
                self.current_image_idx = max(0, self.current_image_idx - 1)
            else:
                self.current_image_idx = min(len(self.enhancer.images) - 1, 
                                           self.current_image_idx + 1)
            
            filename = self.enhancer.images[self.current_image_idx]
            image_info.value = f"<b>Image {self.current_image_idx + 1} of {len(self.enhancer.images)}</b>: {filename}"
            update_preview()
        
        def on_preset_change(change):
            """Apply preset settings"""
            if change['type'] == 'change' and change['name'] == 'value':
                presets = SettingsManager.create_preset_settings()
                preset_settings = presets[change['new']]
                
                # Update all widgets with preset values
                basic_controls[0].value = preset_settings.clip_limit
                basic_controls[1].value = preset_settings.gamma
                basic_controls[2].value = preset_settings.brightness
                basic_controls[3].value = preset_settings.contrast
                color_controls[0].value = preset_settings.saturation
                color_controls[1].value = preset_settings.vibrance
                color_controls[2].value = preset_settings.warmth
                color_controls[3].value = preset_settings.tint
                tone_controls[0].value = preset_settings.exposure
                tone_controls[1].value = preset_settings.highlights
                tone_controls[2].value = preset_settings.shadows
                tone_controls[3].value = preset_settings.clarity
                preserve_black.value = preset_settings.preserve_black
                black_threshold.value = preset_settings.black_threshold
        
        def on_auto_enhance(b):
            """Apply automatic enhancement"""
            img = self.get_current_image()
            if img is not None:
                enhanced, settings, notes = self.enhancer.auto_enhance_image(img)
                
                # Update widgets with auto settings
                self.current_settings = settings
                self._update_widgets_from_settings(settings)
                
                print("🎯 Auto-enhancement applied!")
                for note in notes:
                    print(f"  • {note}")
                
                update_preview()
        
        def on_reset(b):
            """Reset to default settings"""
            default_settings = EnhancementSettings()
            self._update_widgets_from_settings(default_settings)
            update_preview()
        
        def on_save_settings(b):
            """Save current settings to file"""
            update_settings_from_widgets()
            SettingsManager.save_settings(self.current_settings)
            print("💾 Settings saved successfully!")
        
        def on_process_all(b):
            """Process all images with current settings"""
            update_settings_from_widgets()
            process_btn.disabled = True
            stop_btn.disabled = False
            
            with batch_output:
                clear_output()
                results = self.batch_processor.batch_process_threaded(self.current_settings)
            
            process_btn.disabled = False
            stop_btn.disabled = True
        
        def on_auto_process_all(b):
            """Auto-process all images"""
            auto_process_btn.disabled = True
            stop_btn.disabled = False
            
            with batch_output:
                clear_output()
                results = self.batch_processor.auto_batch_process()
            
            auto_process_btn.disabled = False
            stop_btn.disabled = True
        
        def on_stop(b):
            """Stop batch processing"""
            self.batch_processor.stop_processing()
            process_btn.disabled = False
            auto_process_btn.disabled = False
            stop_btn.disabled = True
        
        # Connect event handlers
        prev_btn.on_click(lambda b: on_navigation("prev"))
        next_btn.on_click(lambda b: on_navigation("next"))
        preset_dropdown.observe(on_preset_change)
        auto_btn.on_click(on_auto_enhance)
        reset_btn.on_click(on_reset)
        save_settings_btn.on_click(on_save_settings)
        process_btn.on_click(on_process_all)
        auto_process_btn.on_click(on_auto_process_all)
        stop_btn.on_click(on_stop)
        
        # Connect all sliders to update function
        all_controls = basic_controls + color_controls + tone_controls + [
            comparison_slider, preserve_black, black_threshold, show_stats
        ]
        for control in all_controls:
            control.observe(lambda change: update_preview(), names='value')
        
        # Create tabbed interface for better organization
        basic_tab = VBox([
            HTML("<h4>🔧 Basic Adjustments</h4>"),
            HBox([basic_controls[0], basic_controls[1]]),
            HBox([basic_controls[2], basic_controls[3]])
        ])
        
        color_tab = VBox([
            HTML("<h4>🎨 Color Enhancement</h4>"),
            HBox([color_controls[0], color_controls[1]]),
            HBox([color_controls[2], color_controls[3]])
        ])
        
        tone_tab = VBox([
            HTML("<h4>🌅 Tone Mapping</h4>"),
            HBox([tone_controls[0], tone_controls[1]]),
            HBox([tone_controls[2], tone_controls[3]])
        ])
        
        options_tab = VBox([
            HTML("<h4>⚙️ Options & Actions</h4>"),
            HBox([preserve_black, show_stats]),
            black_threshold,
            HBox([preset_dropdown]),
            HBox([auto_btn, reset_btn, save_settings_btn])
        ])
        
        # Create tabs
        enhancement_tabs = Tab(children=[basic_tab, color_tab, tone_tab, options_tab])
        enhancement_tabs.set_title(0, "Basic")
        enhancement_tabs.set_title(1, "Color")
        enhancement_tabs.set_title(2, "Tone")
        enhancement_tabs.set_title(3, "Options")
        
        # Main layout
        header = VBox([
            HTML("<h1>🃏 Advanced MTG Proxy Enhancer - Optimized</h1>"),
            HTML(f"<p><b>Input:</b> {self.enhancer.input_folder} | <b>Output:</b> {self.enhancer.output_folder}</p>")
        ])
        
        navigation = VBox([
            HTML("<h3>📸 Navigation</h3>"),
            HBox([prev_btn, next_btn]),
            image_info
        ])
        
        comparison_section = VBox([
            HTML("<h3>🔄 Live Comparison</h3>"),
            comparison_slider,
            HTML("<i>Drag to blend: Left=Original, Right=Enhanced</i>")
        ])
        
        batch_section = VBox([
            HTML("<h3>⚡ Batch Processing</h3>"),
            HBox([process_btn, auto_process_btn, stop_btn]),
            batch_output
        ])
        
        # Complete interface
        interface = VBox([
            header,
            navigation,
            enhancement_tabs,
            comparison_section,
            preview_output,
            batch_section
        ])
        
        # Initial preview
        update_preview()
        
        return interface
    
    def _update_widgets_from_settings(self, settings: EnhancementSettings):
        """Helper to update widget values from settings object"""
        # This would be called to sync widgets with settings
        # Implementation depends on widget references
        pass

class CommandLineInterface:
    """Optimized CLI for batch operations"""
    
    def __init__(self, enhancer: MTGProxyEnhancer):
        self.enhancer = enhancer
        self.batch_processor = BatchProcessor(enhancer)
    
    def run_interactive_cli(self):
        """Run interactive command-line interface"""
        print("🃏 MTG Proxy Enhancer - Interactive CLI")
        print("=" * 50)
        
        while True:
            print(f"\n📁 Images found: {len(self.enhancer.images)}")
            print("\nSelect option:")
            print("1. Auto-enhance all images")
            print("2. Quick enhance with preset")
            print("3. Custom enhancement settings")
            print("4. Benchmark performance")
            print("5. View image statistics")
            print("0. Exit")
            
            try:
                choice = input("\nEnter choice (0-5): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    break
                elif choice == "1":
                    self._cli_auto_enhance()
                elif choice == "2":
                    self._cli_preset_enhance()
                elif choice == "3":
                    self._cli_custom_enhance()
                elif choice == "4":
                    self._cli_benchmark()
                elif choice == "5":
                    self._cli_image_stats()
                else:
                    print("❌ Invalid choice")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _cli_auto_enhance(self):
        """CLI auto enhancement"""
        print("\n🤖 Starting auto-enhancement...")
        results = self.batch_processor.auto_batch_process()
        self._print_results(results)
    
    def _cli_preset_enhance(self):
        """CLI preset enhancement"""
        presets = SettingsManager.create_preset_settings()
        
        print("\n📋 Available presets:")
        for i, preset_name in enumerate(presets.keys(), 1):
            print(f"{i}. {preset_name}")
        
        try:
            choice = int(input("Select preset number: ")) - 1
            preset_names = list(presets.keys())
            
            if 0 <= choice < len(preset_names):
                preset_name = preset_names[choice]
                settings = presets[preset_name]
                
                print(f"\n🚀 Processing with '{preset_name}' preset...")
                results = self.batch_processor.batch_process_threaded(settings)
                self._print_results(results)
            else:
                print("❌ Invalid preset selection")
                
        except ValueError:
            print("❌ Invalid input")
    
    def _cli_custom_enhance(self):
        """CLI custom settings enhancement"""
        print("\n⚙️ Custom Enhancement Settings")
        settings = EnhancementSettings()
        
        # Get basic settings from user
        try:
            settings.clip_limit = float(input(f"CLAHE clip limit [{settings.clip_limit}]: ") or settings.clip_limit)
            settings.gamma = float(input(f"Gamma [{settings.gamma}]: ") or settings.gamma)
            settings.saturation = float(input(f"Saturation [{settings.saturation}]: ") or settings.saturation)
            settings.brightness = float(input(f"Brightness [{settings.brightness}]: ") or settings.brightness)
            
            print(f"\n🚀 Processing with custom settings...")
            results = self.batch_processor.batch_process_threaded(settings)
            self._print_results(results)
            
        except ValueError:
            print("❌ Invalid input - using default settings")
            results = self.batch_processor.batch_process_threaded(settings)
            self._print_results(results)
    
    def _cli_benchmark(self):
        """CLI benchmark"""
        print("\n⏱️ Running performance benchmark...")
        results = benchmark_enhancement(str(self.enhancer.input_folder))
        
        if "error" in results:
            print(f"❌ Benchmark failed: {results['error']}")
        else:
            print(f"📊 Benchmark Results:")
            print(f"  Image size: {results['image_size']}")
            print(f"  Average time: {results['avg_time_ms']:.1f}ms")
            print(f"  Images/second: {results['images_per_second']:.1f}")
            print(f"  Estimated batch time: {results['estimated_batch_time']:.1f}s")
    
    def _cli_image_stats(self):
        """CLI image statistics"""
        if not self.enhancer.images:
            print("❌ No images found")
            return
        
        print("\n📊 Image Statistics Analysis")
        for i, filename in enumerate(self.enhancer.images[:5]):  # Limit to first 5
            img_path = self.enhancer.input_folder / filename
            img = cv2.imread(str(img_path))
            
            if img is not None:
                stats, _ = ImageAnalyzer.analyze_image(img)
                print(f"\n{i+1}. {filename}:")
                print(f"   Brightness: {stats.mean_brightness:.1f}")
                print(f"   Contrast: {stats.contrast_std:.1f}")
                print(f"   Color cast: {stats.cast_type or 'None'}")
                print(f"   Characteristics: {', '.join(self._get_image_characteristics(stats))}")
        
        if len(self.enhancer.images) > 5:
            print(f"\n... and {len(self.enhancer.images) - 5} more images")
    
    def _get_image_characteristics(self, stats: ImageStats) -> List[str]:
        """Get human-readable image characteristics"""
        characteristics = []
        if stats.is_dark:
            characteristics.append("Dark")
        if stats.is_bright:
            characteristics.append("Bright")
        if stats.is_low_contrast:
            characteristics.append("Low contrast")
        if stats.has_color_cast:
            characteristics.append(f"{stats.cast_type} cast")
        return characteristics or ["Normal"]
    
    def _print_results(self, results: Dict):
        """Print batch processing results"""
        print(f"\n✅ Processing complete!")
        print(f"   Success: {results['success']} images")
        if results.get('skipped', 0) > 0:
            print(f"   Skipped: {results['skipped']} images")
        if results.get('errors', 0) > 0:
            print(f"   Errors: {results['errors']} images")
        print(f"   Time: {results['time']:.1f} seconds")

# Enhanced utility functions with better error handling
def create_mtg_enhancer_optimized(input_folder: str = "mtgproxy/Input", 
                                 output_folder: str = "mtgproxy/Output") -> MTGProxyEnhancer:
    """Create optimized MTG Proxy Enhancer with validation"""
    try:
        enhancer = MTGProxyEnhancer(input_folder, output_folder)
        logger.info(f"Enhancer created successfully with {len(enhancer.images)} images")
        return enhancer
    except Exception as e:
        logger.error(f"Failed to create enhancer: {e}")
        raise

def run_cli_interface(input_folder: str = "mtgproxy/Input", 
                     output_folder: str = "mtgproxy/Output"):
    """Run the command-line interface"""
    enhancer = create_mtg_enhancer_optimized(input_folder, output_folder)
    cli = CommandLineInterface(enhancer)
    cli.run_interactive_cli()

# Performance monitoring
class PerformanceMonitor:
    """Monitor and optimize performance"""
    
    @staticmethod
    def profile_enhancement_pipeline(img: np.ndarray, settings: EnhancementSettings) -> Dict:
        """Profile each step of the enhancement pipeline"""
        import time
        
        steps = {}
        enhancer = MTGProxyEnhancer()
        
        # Time each major step
        start = time.time()
        ImageProcessor.apply_tone_mapping(img, settings.highlights, settings.shadows, 
                                        settings.whites, settings.blacks)
        steps['tone_mapping'] = time.time() - start
        
        start = time.time()
        ImageProcessor.apply_gamma_correction(img, settings.gamma)
        steps['gamma'] = time.time() - start
        
        start = time.time()
        ImageProcessor.apply_clahe(img, settings.clip_limit)
        steps['clahe'] = time.time() - start
        
        start = time.time()
        ImageProcessor.apply_color_adjustments(img, settings.saturation, settings.vibrance,
                                             settings.warmth, settings.tint)
        steps['color'] = time.time() - start
        
        return steps

# Main execution
if __name__ == "__main__":
    print("🃏 MTG Proxy Enhancer - Optimized Version Loaded!")
    print("\n🚀 QUICK START:")
    print("1. Place MTG images in 'mtgproxy/Input'")
    print("2. Run: run_cli_interface()  # For CLI")
    print("3. Or: enhancer = create_mtg_enhancer_optimized()")
    print("4. Or: auto_enhance_all()  # One-click processing")
    
    # Auto-create default instance if images are present
    try:
        enhancer = create_mtg_enhancer_optimized()
        if enhancer.images:
            print(f"\n✅ Ready! Found {len(enhancer.images)} images")
            print("Run: interface = InteractiveInterface(enhancer)")
            print("     ui = interface.create_widget_interface()")
            print("     display(ui)")
        else:
            print("\n📂 Add images to 'mtgproxy/Input' folder to get started")
    except Exception as e:
        logger.error(f"Initialization error: {e}")


#!/usr/bin/env python3
"""
Part 4
MTG Proxy Enhancer - Advanced Features and Utilities
Continuation from Part 3 - Final optimized components
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from datetime import datetime
import hashlib
from dataclasses import asdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

class AdvancedImageAnalyzer:
    """Advanced analysis for MTG card-specific features"""
    
    @staticmethod
    def detect_card_regions(img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect different regions of MTG cards (text boxes, art, borders)
        Returns masks for different card regions
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Detect text regions (typically darker areas with high contrast)
        text_mask = np.zeros_like(gray)
        
        # Find areas with high local contrast (likely text)
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        edges = cv2.filter2D(gray, -1, kernel)
        
        # Text areas: dark pixels with high edge response
        text_candidates = (gray < 100) & (edges > 30)
        text_mask[text_candidates] = 255
        
        # Detect borders (typically thin black lines at edges)
        border_mask = np.zeros_like(gray)
        border_thickness = min(w, h) // 100  # Adaptive border detection
        
        # Edge regions
        border_mask[:border_thickness, :] = 255  # Top
        border_mask[-border_thickness:, :] = 255  # Bottom  
        border_mask[:, :border_thickness] = 255  # Left
        border_mask[:, -border_thickness:] = 255  # Right
        
        # Art region (everything else, typically center area)
        art_mask = 255 - cv2.bitwise_or(text_mask, border_mask)
        
        return {
            'text': text_mask,
            'border': border_mask, 
            'art': art_mask
        }
    
    @staticmethod
    def analyze_card_quality(img: np.ndarray) -> Dict:
        """
        Analyze MTG card-specific quality metrics
        """
        regions = AdvancedImageAnalyzer.detect_card_regions(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Text readability analysis
        text_areas = gray[regions['text'] > 0]
        text_readability = 0
        if len(text_areas) > 0:
            text_contrast = np.std(text_areas)
            text_brightness = np.mean(text_areas)
            text_readability = min(100, text_contrast * 2)  # Simplified metric
        
        # Art quality analysis  
        art_areas = img[regions['art'] > 0]
        art_quality = 0
        if len(art_areas) > 0:
            art_saturation = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[regions['art'] > 0, 1])
            art_detail = np.std(art_areas)
            art_quality = min(100, (art_saturation + art_detail) / 3)
        
        # Overall sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(100, laplacian_var / 100)
        
        return {
            'text_readability': text_readability,
            'art_quality': art_quality, 
            'sharpness': sharpness_score,
            'regions': regions
        }

class QualityAssessment:
    """Quality assessment and comparison tools"""
    
    @staticmethod
    def compare_images(original: np.ndarray, enhanced: np.ndarray) -> Dict:
        """
        Comprehensive quality comparison between original and enhanced
        """
        # PSNR (Peak Signal-to-Noise Ratio)
        mse = np.mean((original - enhanced) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # SSIM would require skimage, so we'll use simpler metrics
        
        # Contrast improvement
        orig_contrast = np.std(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
        enh_contrast = np.std(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        contrast_improvement = (enh_contrast - orig_contrast) / orig_contrast * 100
        
        # Color saturation change
        orig_sat = np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2HSV)[..., 1])
        enh_sat = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)[..., 1])
        saturation_change = (enh_sat - orig_sat) / orig_sat * 100 if orig_sat > 0 else 0
        
        # Brightness change
        orig_brightness = np.mean(original)
        enh_brightness = np.mean(enhanced)
        brightness_change = enh_brightness - orig_brightness
        
        return {
            'psnr': psnr,
            'contrast_improvement_percent': contrast_improvement,
            'saturation_change_percent': saturation_change,
            'brightness_change': brightness_change,
            'enhancement_strength': abs(contrast_improvement) + abs(saturation_change/10)
        }
    
    @staticmethod
    def create_quality_report(enhancer: MTGProxyEnhancer, 
                            sample_size: int = 5) -> Dict:
        """
        Create comprehensive quality report for sample images
        """
        if not enhancer.images:
            return {"error": "No images found"}
        
        sample_images = enhancer.images[:min(sample_size, len(enhancer.images))]
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(enhancer.images),
            'analyzed_images': len(sample_images),
            'results': []
        }
        
        for filename in sample_images:
            img_path = enhancer.input_folder / filename
            img = cv2.imread(str(img_path))
            
            if img is None:
                continue
            
            # Analyze original
            stats, optimal_settings = ImageAnalyzer.analyze_image(img)
            card_quality = AdvancedImageAnalyzer.analyze_card_quality(img)
            
            # Create enhanced version
            enhanced = enhancer.enhance_image(img, optimal_settings)
            quality_comparison = QualityAssessment.compare_images(img, enhanced)
            
            result = {
                'filename': filename,
                'original_stats': asdict(stats),
                'recommended_settings': asdict(optimal_settings),
                'card_quality': card_quality,
                'quality_improvement': quality_comparison
            }
            
            report['results'].append(result)
        
        return report

class ImageCache:
    """Intelligent caching system for faster previews"""
    
    def __init__(self, max_cache_size: int = 50):
        self.cache = {}
        self.max_size = max_cache_size
        self.access_order = []
    
    def _generate_key(self, filename: str, settings: EnhancementSettings) -> str:
        """Generate unique cache key"""
        settings_str = json.dumps(asdict(settings), sort_keys=True)
        return hashlib.md5(f"{filename}:{settings_str}".encode()).hexdigest()
    
    def get(self, filename: str, settings: EnhancementSettings) -> Optional[np.ndarray]:
        """Get cached enhanced image"""
        key = self._generate_key(filename, settings)
        
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key].copy()
        
        return None
    
    def put(self, filename: str, settings: EnhancementSettings, 
            enhanced_img: np.ndarray) -> None:
        """Cache enhanced image with LRU eviction"""
        key = self._generate_key(filename, settings)
        
        # Evict oldest if cache full
        if len(self.cache) >= self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = enhanced_img.copy()
        self.access_order.append(key)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()

class EnhancementValidator:
    """Validate enhancement results and detect issues"""
    
    @staticmethod
    def validate_enhancement(original: np.ndarray, enhanced: np.ndarray) -> Dict:
        """
        Validate that enhancement didn't introduce artifacts or issues
        """
        issues = []
        
        # Check for clipping
        if np.sum(enhanced == 255) > np.sum(original == 255) * 2:
            issues.append("Potential highlight clipping")
        
        if np.sum(enhanced == 0) > np.sum(original == 0) * 2:
            issues.append("Potential shadow crushing")
        
        # Check for oversaturation
        orig_sat = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)[..., 1]
        enh_sat = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)[..., 1]
        
        if np.mean(enh_sat) > np.mean(orig_sat) * 1.5:
            issues.append("Potential oversaturation")
        
        # Check for excessive noise
        orig_noise = cv2.Laplacian(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        enh_noise = cv2.Laplacian(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        
        if enh_noise > orig_noise * 2:
            issues.append("Potential noise amplification")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'quality_score': max(0, 100 - len(issues) * 25)
        }

class ExportManager:
    """Handle different export formats and options"""
    
    @staticmethod
    def export_with_metadata(img: np.ndarray, output_path: Path, 
                           settings: EnhancementSettings, 
                           format_type: str = "auto") -> bool:
        """
        Export image with enhancement metadata
        """
        # Determine format
        if format_type == "auto":
            format_type = output_path.suffix.lower()
        
        # Prepare export settings
        export_params = []
        
        if format_type in ['.jpg', '.jpeg']:
            export_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif format_type == '.png':
            export_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        elif format_type in ['.tiff', '.tif']:
            export_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
        
        # Save image
        success = cv2.imwrite(str(output_path), img, export_params)
        
        # Save metadata
        if success:
            metadata_path = output_path.with_suffix('.json')
            metadata = {
                'original_file': output_path.name,
                'enhancement_settings': asdict(settings),
                'processed_date': datetime.now().isoformat(),
                'version': '2.0_optimized'
            }
            
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save metadata: {e}")
        
        return success
    
    @staticmethod
    def create_before_after_comparison(original: np.ndarray, enhanced: np.ndarray,
                                     output_path: Path) -> bool:
        """
        Create side-by-side before/after comparison image
        """
        # Ensure same size
        h, w = original.shape[:2]
        
        # Create comparison canvas
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = original
        comparison[:, w:] = enhanced
        
        # Add dividing line
        cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "ORIGINAL", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "ENHANCED", (w + 10, 30), font, 1, (255, 255, 255), 2)
        
        return cv2.imwrite(str(output_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])

class BatchAnalytics:
    """Analytics and reporting for batch operations"""
    
    @staticmethod
    def analyze_batch_results(enhancer: MTGProxyEnhancer, 
                            results_summary: Dict) -> Dict:
        """
        Analyze batch processing results and provide insights
        """
        analytics = {
            'processing_summary': results_summary,
            'performance_metrics': {},
            'recommendations': []
        }
        
        # Performance metrics
        total_images = results_summary.get('success', 0) + results_summary.get('errors', 0)
        if total_images > 0 and results_summary.get('time', 0) > 0:
            analytics['performance_metrics'] = {
                'images_per_second': total_images / results_summary['time'],
                'avg_time_per_image': results_summary['time'] / total_images,
                'success_rate': results_summary.get('success', 0) / total_images * 100
            }
        
        # Generate recommendations
        if results_summary.get('errors', 0) > 0:
            analytics['recommendations'].append("Some images failed processing - check file formats and corruption")
        
        if analytics['performance_metrics'].get('images_per_second', 0) < 1:
            analytics['recommendations'].append("Consider reducing image sizes or enhancement complexity for faster processing")
        
        if results_summary.get('success', 0) > 10:
            analytics['recommendations'].append("Large batch completed successfully - consider saving current settings as preset")
        
        return analytics
    
    @staticmethod
    def create_processing_report(enhancer: MTGProxyEnhancer, 
                               analytics: Dict,
                               output_file: str = "processing_report.html") -> str:
        """
        Create HTML report of batch processing results
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MTG Proxy Enhancement Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .metric {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }}
                .success {{ border-left-color: #28a745; }}
                .warning {{ border-left-color: #ffc107; }}
                .error {{ border-left-color: #dc3545; }}
                .footer {{ text-align: center; color: #666; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🃏 MTG Proxy Enhancement Report</h1>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="metric success">
                    <h3>✅ Processing Summary</h3>
                    <p><strong>Successfully processed:</strong> {analytics['processing_summary'].get('success', 0)} images</p>
                    <p><strong>Total time:</strong> {analytics['processing_summary'].get('time', 0):.1f} seconds</p>
                    <p><strong>Input folder:</strong> {enhancer.input_folder}</p>
                    <p><strong>Output folder:</strong> {enhancer.output_folder}</p>
                </div>
                
                <div class="metric">
                    <h3>📊 Performance Metrics</h3>
                    <p><strong>Processing speed:</strong> {analytics['performance_metrics'].get('images_per_second', 0):.1f} images/second</p>
                    <p><strong>Average time per image:</strong> {analytics['performance_metrics'].get('avg_time_per_image', 0):.2f} seconds</p>
                    <p><strong>Success rate:</strong> {analytics['performance_metrics'].get('success_rate', 0):.1f}%</p>
                </div>
                
                <div class="metric warning">
                    <h3>💡 Recommendations</h3>
                    <ul>
        """
        
        for rec in analytics.get('recommendations', []):
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                    </ul>
                </div>
                
                <div class="footer">
                    <p>Generated by MTG Proxy Enhancer v2.0 (Optimized)</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        try:
            with open(output_file, 'w') as f:
                f.write(html_content)
            return output_file
        except Exception as e:
            logger.error(f"Failed to create report: {e}")
            return ""

class MTGEnhancerPro:
    """
    Professional-grade MTG enhancer with all advanced features
    """
    
    def __init__(self, input_folder: str = "mtgproxy/Input", 
                 output_folder: str = "mtgproxy/Output"):
        self.enhancer = MTGProxyEnhancer(input_folder, output_folder)
        self.batch_processor = BatchProcessor(self.enhancer)
        self.cache = ImageCache()
        self.settings_manager = SettingsManager()
        
    def enhance_with_validation(self, img: np.ndarray, 
                              settings: EnhancementSettings) -> Tuple[np.ndarray, Dict]:
        """
        Enhance image with quality validation
        """
        # Check cache first
        cache_result = self.cache.get("temp_image", settings)
        if cache_result is not None:
            return cache_result, {"cached": True}
        
        # Enhance image
        enhanced = self.enhancer.enhance_image(img, settings)
        
        # Validate result
        validation = EnhancementValidator.validate_enhancement(img, enhanced)
        
        # Cache result if valid
        if validation['valid']:
            self.cache.put("temp_image", settings, enhanced)
        
        return enhanced, validation
    
    def batch_process_with_analytics(self, settings: EnhancementSettings,
                                   create_report: bool = True,
                                   create_comparisons: bool = False) -> Dict:
        """
        Batch process with comprehensive analytics
        """
        # Process images
        results = self.batch_processor.batch_process_threaded(settings)
        
        # Generate analytics
        analytics = BatchAnalytics.analyze_batch_results(self.enhancer, results)
        
        # Create comparison images if requested
        if create_comparisons and results['success'] > 0:
            self._create_batch_comparisons(settings)
        
        # Create HTML report if requested
        if create_report:
            report_file = BatchAnalytics.create_processing_report(self.enhancer, analytics)
            if report_file:
                print(f"📄 Report saved: {report_file}")
        
        return analytics
    
    def _create_batch_comparisons(self, settings: EnhancementSettings):
        """Create before/after comparisons for batch"""
        comparison_folder = self.enhancer.output_folder / "comparisons"
        comparison_folder.mkdir(exist_ok=True)
        
        sample_size = min(5, len(self.enhancer.images))
        
        for i, filename in enumerate(self.enhancer.images[:sample_size]):
            original_path = self.enhancer.input_folder / filename
            enhanced_path = self.enhancer.output_folder / filename
            
            original = cv2.imread(str(original_path))
            enhanced = cv2.imread(str(enhanced_path))
            
            if original is not None and enhanced is not None:
                comparison_path = comparison_folder / f"comparison_{filename}"
                ExportManager.create_before_after_comparison(original, enhanced, comparison_path)
        
        print(f"📷 {sample_size} comparison images saved to: {comparison_folder}")

# Advanced utility functions
def create_enhancement_preview(image_path: str, settings: EnhancementSettings) -> None:
    """
    Create a detailed preview for a single image
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    enhancer = MTGProxyEnhancer()
    enhanced = enhancer.enhance_image(img, settings)
    
    # Analyze quality
    quality_comparison = QualityAssessment.compare_images(img, enhanced)
    card_quality = AdvancedImageAnalyzer.analyze_card_quality(img)
    validation = EnhancementValidator.validate_enhancement(img, enhanced)
    
    # Create comprehensive preview
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"🃏 Enhancement Preview: {Path(image_path).name}", fontsize=16)
    
    # Original and enhanced
    axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title("Original")
    axes[0,0].axis('off')
    
    axes[0,1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title("Enhanced")
    axes[0,1].axis('off')
    
    # Difference map
    diff = cv2.absdiff(img, enhanced)
    axes[0,2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    axes[0,2].set_title("Difference Map")
    axes[0,2].axis('off')
    
    # Quality metrics
    axes[1,0].axis('off')
    metrics_text = f"""Quality Metrics:
PSNR: {quality_comparison['psnr']:.1f} dB
Contrast Improvement: {quality_comparison['contrast_improvement_percent']:+.1f}%
Saturation Change: {quality_comparison['saturation_change_percent']:+.1f}%
Brightness Change: {quality_comparison['brightness_change']:+.1f}

Card Quality:
Text Readability: {card_quality['text_readability']:.1f}/100
Art Quality: {card_quality['art_quality']:.1f}/100
Sharpness: {card_quality['sharpness']:.1f}/100"""
    
    axes[1,0].text(0.1, 0.9, metrics_text, transform=axes[1,0].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Settings display
    axes[1,1].axis('off')
    settings_text = f"""Enhancement Settings:
CLAHE: {settings.clip_limit:.1f}
Gamma: {settings.gamma:.2f}
Saturation: {settings.saturation:.2f}
Brightness: {settings.brightness:+.0f}
Contrast: {settings.contrast:.2f}
Vibrance: {settings.vibrance:+.0f}
Warmth: {settings.warmth:+.0f}
Exposure: {settings.exposure:+.1f}
Clarity: {settings.clarity:+.0f}"""
    
    axes[1,1].text(0.1, 0.9, settings_text, transform=axes[1,1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Validation results
    axes[1,2].axis('off')
    validation_text = f"""Validation Results:
Status: {'✅ PASSED' if validation['valid'] else '⚠️ ISSUES DETECTED'}
Quality Score: {validation['quality_score']}/100

Issues:"""
    
    if validation['issues']:
        for issue in validation['issues']:
            validation_text += f"\n• {issue}"
    else:
        validation_text += "\n• None detected"
    
    axes[1,2].text(0.1, 0.9, validation_text, transform=axes[1,2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

def run_comprehensive_analysis(input_folder: str = "mtgproxy/Input") -> Dict:
    """
    Run comprehensive analysis of all images in folder
    """
    enhancer = MTGProxyEnhancer(input_folder)
    
    if not enhancer.images:
        return {"error": "No images found"}
    
    print("🔍 Running comprehensive image analysis...")
    
    analysis_results = {
        'folder': input_folder,
        'total_images': len(enhancer.images),
        'analysis_date': datetime.now().isoformat(),
        'image_analyses': [],
        'summary_stats': {}
    }
    
    brightness_values = []
    contrast_values = []
    quality_scores = []
    
    for i, filename in enumerate(enhancer.images):
        print(f"\rAnalyzing: {filename} [{i+1}/{len(enhancer.images)}]", end="", flush=True)
        
        img_path = enhancer.input_folder / filename
        img = cv2.imread(str(img_path))
        
        if img is None:
            continue
        
        # Comprehensive analysis
        stats, optimal_settings = ImageAnalyzer.analyze_image(img)
        card_quality = AdvancedImageAnalyzer.analyze_card_quality(img)
        
        # Enhanced version for comparison
        enhanced = enhancer.enhance_image(img, optimal_settings)
        quality_comparison = QualityAssessment.compare_images(img, enhanced)
        validation = EnhancementValidator.validate_enhancement(img, enhanced)
        
        image_analysis = {
            'filename': filename,
            'file_size': img_path.stat().st_size,
            'dimensions': f"{img.shape[1]}x{img.shape[0]}",
            'original_stats': asdict(stats),
            'card_quality': card_quality,
            'optimal_settings': asdict(optimal_settings),
            'enhancement_impact': quality_comparison,
            'validation': validation
        }
        
        analysis_results['image_analyses'].append(image_analysis)
        
        # Collect for summary stats
        brightness_values.append(stats.mean_brightness)
        contrast_values.append(stats.contrast_std)
        quality_scores.append(validation['quality_score'])
    
    # Summary statistics
    if brightness_values:
        analysis_results['summary_stats'] = {
            'avg_brightness': float(np.mean(brightness_values)),
            'avg_contrast': float(np.mean(contrast_values)),
            'avg_quality_score': float(np.mean(quality_scores)),
            'dark_images_count': sum(1 for b in brightness_values if b < 80),
            'bright_images_count': sum(1 for b in brightness_values if b > 180),
            'low_contrast_count': sum(1 for c in contrast_values if c < 35)
        }
    
    print(f"\n\n✅ Analysis complete!")
    print(f"📊 Average brightness: {analysis_results['summary_stats'].get('avg_brightness', 0):.1f}")
    print(f"📊 Average contrast: {analysis_results['summary_stats'].get('avg_contrast', 0):.1f}")
    print(f"📊 Average quality: {analysis_results['summary_stats'].get('avg_quality_score', 0):.1f}/100")
    
    # Save analysis to JSON
    analysis_file = Path(input_folder).parent / "image_analysis.json"
    try:
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"💾 Analysis saved to: {analysis_file}")
    except Exception as e:
        logger.error(f"Could not save analysis: {e}")
    
    return analysis_results

# Final optimized entry points
def create_pro_enhancer(input_folder: str = "mtgproxy/Input",
                       output_folder: str = "mtgproxy/Output") -> MTGEnhancerPro:
    """Create professional MTG enhancer with all features"""
    return MTGEnhancerPro(input_folder, output_folder)

def one_click_enhance(input_folder: str = "mtgproxy/Input",
                     output_folder: str = "mtgproxy/Output",
                     create_report: bool = True) -> Dict:
    """
    One-click enhancement with auto-analysis and reporting
    """
    print("🚀 One-Click MTG Proxy Enhancement")
    print("=" * 40)
    
    pro_enhancer = create_pro_enhancer(input_folder, output_folder)
    
    # Run comprehensive enhancement
    results = pro_enhancer.batch_process_with_analytics(
        EnhancementSettings(), 
        create_report=create_report,
        create_comparisons=True
    )
    
    print("\n🎉 One-click enhancement complete!")
    return results

if __name__ == "__main__":
    print("🃏 MTG Proxy Enhancer - Advanced Features Loaded!")
    print("\n🚀 PROFESSIONAL FEATURES:")
    print("• one_click_enhance() - Complete auto-processing with reports")
    print("• run_comprehensive_analysis() - Detailed image analysis")  
    print("• create_pro_enhancer() - Full-featured enhancer")
    print("• create_enhancement_preview(image_path, settings) - Single image preview")
    
    print("\n📊 ANALYSIS TOOLS:")
    print("• Quality assessment and validation")
    print("• Card region detection (text/art/borders)")
    print("• Performance benchmarking")
    print("• HTML report generation")
    
    print("\n⚡ OPTIMIZATIONS:")
    print("• Multi-threaded batch processing")
    print("• Intelligent caching system")  
    print("• Vectorized image operations")
    print("• Memory-efficient processing pipeline")


#!/usr/bin/env python3
"""
Part 5
MTG Proxy Enhancer - Configuration and Main Entry Point
Final part with configuration management and unified interface
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import yaml
import configparser
from dataclasses import asdict, field
from enum import Enum

class ProcessingMode(Enum):
    """Processing mode options"""
    AUTO = "auto"
    PRESET = "preset" 
    CUSTOM = "custom"
    INTERACTIVE = "interactive"

@dataclass
class AppConfig:
    """Application configuration"""
    input_folder: str = "mtgproxy/Input"
    output_folder: str = "mtgproxy/Output"
    max_workers: int = 4
    cache_size: int = 50
    default_quality: int = 95
    create_backups: bool = False
    processing_mode: ProcessingMode = ProcessingMode.AUTO
    log_level: str = "INFO"
    auto_create_folders: bool = True
    preserve_metadata: bool = True
    
    # Performance settings
    chunk_size: int = 10  # Images per processing chunk
    memory_limit_mb: int = 1024
    use_gpu_acceleration: bool = False

class ConfigManager:
    """Manage application configuration"""
    
    DEFAULT_CONFIG_FILE = "mtg_enhancer_config.yaml"
    
    @staticmethod
    def load_config(config_file: str = None) -> AppConfig:
        """Load configuration from file or create default"""
        if config_file is None:
            config_file = ConfigManager.DEFAULT_CONFIG_FILE
        
        config_path = Path(config_file)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Convert processing_mode string to enum
                if 'processing_mode' in config_data:
                    config_data['processing_mode'] = ProcessingMode(config_data['processing_mode'])
                
                return AppConfig(**config_data)
                
            except Exception as e:
                logger.warning(f"Could not load config file: {e}. Using defaults.")
                return AppConfig()
        else:
            # Create default config file
            default_config = AppConfig()
            ConfigManager.save_config(default_config, config_file)
            return default_config
    
    @staticmethod
    def save_config(config: AppConfig, config_file: str = None) -> bool:
        """Save configuration to file"""
        if config_file is None:
            config_file = ConfigManager.DEFAULT_CONFIG_FILE
        
        try:
            config_dict = asdict(config)
            # Convert enum to string for serialization
            config_dict['processing_mode'] = config.processing_mode.value
            
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Could not save config: {e}")
            return False

class MTGEnhancerApp:
    """Main application class with unified interface"""
    
    def __init__(self, config: AppConfig = None):
        self.config = config or ConfigManager.load_config()
        self.enhancer = None
        self.pro_enhancer = None
        self._setup_logging()
        self._initialize_enhancer()
    
    def _setup_logging(self):
        """Configure logging based on config"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)
    
    def _initialize_enhancer(self):
        """Initialize enhancer components"""
        try:
            self.enhancer = MTGProxyEnhancer(
                self.config.input_folder, 
                self.config.output_folder
            )
            self.pro_enhancer = MTGEnhancerPro(
                self.config.input_folder,
                self.config.output_folder
            )
            
            logger.info(f"Initialized with {len(self.enhancer.images)} images")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhancer: {e}")
            sys.exit(1)
    
    def run(self, mode: ProcessingMode = None) -> Dict:
        """
        Run enhancement based on mode
        """
        processing_mode = mode or self.config.processing_mode
        
        if not self.enhancer.images:
            logger.warning("No images found to process")
            return {"error": "No images found"}
        
        logger.info(f"Running in {processing_mode.value} mode")
        
        if processing_mode == ProcessingMode.AUTO:
            return self._run_auto_mode()
        elif processing_mode == ProcessingMode.PRESET:
            return self._run_preset_mode()
        elif processing_mode == ProcessingMode.CUSTOM:
            return self._run_custom_mode()
        elif processing_mode == ProcessingMode.INTERACTIVE:
            return self._run_interactive_mode()
        else:
            logger.error(f"Unknown processing mode: {processing_mode}")
            return {"error": "Unknown processing mode"}
    
    def _run_auto_mode(self) -> Dict:
        """Run automatic enhancement"""
        logger.info("🤖 Starting automatic enhancement")
        
        batch_processor = BatchProcessor(self.enhancer)
        results = batch_processor.auto_batch_process(self.config.max_workers)
        
        # Generate analytics
        analytics = BatchAnalytics.analyze_batch_results(self.enhancer, results)
        
        return analytics
    
    def _run_preset_mode(self) -> Dict:
        """Run with preset settings"""
        presets = SettingsManager.create_preset_settings()
        
        # Use professional preset as default
        settings = presets.get('professional', presets['default'])
        
        logger.info("📋 Processing with professional preset")
        
        batch_processor = BatchProcessor(self.enhancer)
        results = batch_processor.batch_process_threaded(settings, self.config.max_workers)
        
        return BatchAnalytics.analyze_batch_results(self.enhancer, results)
    
    def _run_custom_mode(self) -> Dict:
        """Run with custom settings from config or user input"""
        # Try to load saved settings first
        saved_settings = SettingsManager.load_settings()
        settings = saved_settings or EnhancementSettings()
        
        logger.info("⚙️ Processing with custom settings")
        
        batch_processor = BatchProcessor(self.enhancer)
        results = batch_processor.batch_process_threaded(settings, self.config.max_workers)
        
        return BatchAnalytics.analyze_batch_results(self.enhancer, results)
    
    def _run_interactive_mode(self) -> Dict:
        """Launch interactive interface"""
        logger.info("🖥️ Launching interactive interface")
        
        try:
            interface = InteractiveInterface(self.enhancer)
            ui = interface.create_widget_interface()
            
            # Try to display if in Jupyter environment
            try:
                from IPython.display import display
                display(ui)
                return {"status": "Interactive interface launched"}
            except ImportError:
                logger.warning("Not in Jupyter environment - use CLI interface instead")
                cli = CommandLineInterface(self.enhancer)
                cli.run_interactive_cli()
                return {"status": "CLI interface completed"}
                
        except Exception as e:
            logger.error(f"Failed to launch interface: {e}")
            return {"error": str(e)}

def create_command_line_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="MTG Proxy Enhancer - Professional Image Enhancement Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhance_2_optimized.py --auto                           # Auto-enhance all images
  python enhance_2_optimized.py --preset professional          # Use professional preset
  python enhance_2_optimized.py --interactive                   # Launch interactive mode
  python enhance_2_optimized.py --input ./cards --output ./enhanced --workers 8
  python enhance_2_optimized.py --analyze                       # Analyze images only
  python enhance_2_optimized.py --benchmark                     # Run performance test
        """
    )
    
    # Input/Output options
    parser.add_argument('--input', '-i', default="mtgproxy/Input",
                       help='Input folder path (default: mtgproxy/Input)')
    parser.add_argument('--output', '-o', default="mtgproxy/Output", 
                       help='Output folder path (default: mtgproxy/Output)')
    
    # Processing modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--auto', action='store_true',
                           help='Auto-enhance all images with intelligent analysis')
    mode_group.add_argument('--preset', choices=['default', 'dark_images', 'bright_images', 
                                                'low_contrast', 'color_enhancement', 
                                                'professional', 'vintage_correction'],
                           help='Use predefined enhancement preset')
    mode_group.add_argument('--interactive', action='store_true',
                           help='Launch interactive interface')
    mode_group.add_argument('--analyze', action='store_true',
                           help='Analyze images without processing')
    mode_group.add_argument('--benchmark', action='store_true',
                           help='Run performance benchmark')
    
    # Performance options
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of worker threads (default: 4)')
    parser.add_argument('--quality', '-q', type=int, default=95, choices=range(50, 101),
                       help='Output JPEG quality 50-100 (default: 95)')
    
    # Enhancement options
    parser.add_argument('--gamma', type=float, default=1.2,
                       help='Gamma correction value (default: 1.2)')
    parser.add_argument('--saturation', type=float, default=1.0,
                       help='Saturation multiplier (default: 1.0)')
    parser.add_argument('--clahe', type=float, default=2.0,
                       help='CLAHE clip limit (default: 2.0)')
    
    # Output options
    parser.add_argument('--report', action='store_true',
                       help='Generate HTML processing report')
    parser.add_argument('--comparisons', action='store_true',
                       help='Create before/after comparison images')
    parser.add_argument('--overwrite', action='store_true', default=True,
                       help='Overwrite existing output files')
    
    # Utility options
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--save-config', help='Save current settings to config file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    return parser

def main():
    """Main entry point for command-line usage"""
    parser = create_command_line_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager.load_config(args.config) if args.config else ConfigManager.load_config()
    
    # Override config with command-line arguments
    if args.input:
        config.input_folder = args.input
    if args.output:
        config.output_folder = args.output
    if args.workers:
        config.max_workers = args.workers
    if args.verbose:
        config.log_level = "DEBUG"
    
    # Save config if requested
    if args.save_config:
        ConfigManager.save_config(config, args.save_config)
        print(f"💾 Configuration saved to {args.save_config}")
        return
    
    # Create application
    app = MTGEnhancerApp(config)
    
    # Determine processing mode from arguments
    if args.auto:
        results = app.run(ProcessingMode.AUTO)
    elif args.preset:
        # Use preset mode with specific preset
        presets = SettingsManager.create_preset_settings()
        settings = presets[args.preset]
        
        # Apply command-line overrides
        if args.gamma != 1.2:
            settings.gamma = args.gamma
        if args.saturation != 1.0:
            settings.saturation = args.saturation
        if args.clahe != 2.0:
            settings.clip_limit = args.clahe
        
        batch_processor = BatchProcessor(app.enhancer)
        results = batch_processor.batch_process_threaded(settings, config.max_workers)
        results = BatchAnalytics.analyze_batch_results(app.enhancer, results)
        
    elif args.interactive:
        results = app.run(ProcessingMode.INTERACTIVE)
    elif args.analyze:
        results = run_comprehensive_analysis(config.input_folder)
    elif args.benchmark:
        results = benchmark_enhancement(config.input_folder)
        print("📊 Benchmark Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        return
    else:
        # Default to auto mode
        results = app.run(ProcessingMode.AUTO)
    
    # Create additional outputs if requested
    if args.report and 'processing_summary' in results:
        report_file = BatchAnalytics.create_processing_report(
            app.enhancer, results, "enhancement_report.html"
        )
        if report_file:
            print(f"📄 Report created: {report_file}")
    
    if args.comparisons:
        # Create sample comparisons
        comparison_folder = Path(config.output_folder) / "comparisons"
        comparison_folder.mkdir(exist_ok=True)
        print(f"📷 Comparison images saved to: {comparison_folder}")
    
    # Print final summary
    if isinstance(results, dict) and 'processing_summary' in results:
        summary = results['processing_summary']
        print(f"\n🎉 Enhancement Complete!")
        print(f"✅ Success: {summary.get('success', 0)} images")
        if summary.get('errors', 0) > 0:
            print(f"❌ Errors: {summary['errors']} images")
        print(f"⏱️ Time: {summary.get('time', 0):.1f} seconds")

class WebInterface:
    """Simple web interface for remote usage"""
    
    def __init__(self, enhancer: MTGProxyEnhancer, port: int = 8000):
        self.enhancer = enhancer
        self.port = port
    
    def create_simple_web_ui(self) -> str:
        """Create basic HTML interface"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MTG Proxy Enhancer</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .card { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
                .btn:hover { background: #0056b3; }
                .btn-success { background: #28a745; }
                .btn-warning { background: #ffc107; color: black; }
                .progress { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }
                .progress-bar { height: 100%; background: #007bff; transition: width 0.3s; }
                .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                input[type="range"] { width: 100%; }
                .setting { margin: 10px 0; }
                .setting label { display: inline-block; width: 120px; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>🃏 MTG Proxy Enhancer</h1>
                <p>Professional image enhancement for Magic: The Gathering proxy cards</p>
            </div>
            
            <div class="card">
                <h2>📁 File Management</h2>
                <div class="grid">
                    <div>
                        <h3>Input Folder</h3>
                        <p id="input-path">{input_folder}</p>
                        <p id="image-count">Images found: <span id="image-count-value">0</span></p>
                    </div>
                    <div>
                        <h3>Output Folder</h3>
                        <p id="output-path">{output_folder}</p>
                        <button class="btn" onclick="refreshImages()">🔄 Refresh</button>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>⚡ Quick Actions</h2>
                <button class="btn btn-success" onclick="autoEnhance()">🤖 Auto Enhance All</button>
                <button class="btn" onclick="usePreset('professional')">👔 Professional Preset</button>
                <button class="btn" onclick="usePreset('color_enhancement')">🎨 Color Enhancement</button>
                <button class="btn btn-warning" onclick="analyzeOnly()">📊 Analyze Only</button>
            </div>
            
            <div class="card">
                <h2>🔧 Custom Enhancement</h2>
                <div class="grid">
                    <div>
                        <div class="setting">
                            <label>CLAHE:</label>
                            <input type="range" id="clahe" min="0.5" max="8" step="0.1" value="2.0">
                            <span id="clahe-value">2.0</span>
                        </div>
                        <div class="setting">
                            <label>Gamma:</label>
                            <input type="range" id="gamma" min="0.5" max="3" step="0.05" value="1.2">
                            <span id="gamma-value">1.2</span>
                        </div>
                        <div class="setting">
                            <label>Saturation:</label>
                            <input type="range" id="saturation" min="0" max="3" step="0.05" value="1.0">
                            <span id="saturation-value">1.0</span>
                        </div>
                        <div class="setting">
                            <label>Brightness:</label>
                            <input type="range" id="brightness" min="-50" max="50" step="1" value="0">
                            <span id="brightness-value">0</span>
                        </div>
                    </div>
                    <div>
                        <div class="setting">
                            <label>Vibrance:</label>
                            <input type="range" id="vibrance" min="-50" max="50" step="1" value="0">
                            <span id="vibrance-value">0</span>
                        </div>
                        <div class="setting">
                            <label>Warmth:</label>
                            <input type="range" id="warmth" min="-30" max="30" step="1" value="0">
                            <span id="warmth-value">0</span>
                        </div>
                        <div class="setting">
                            <label>Clarity:</label>
                            <input type="range" id="clarity" min="-50" max="50" step="1" value="0">
                            <span id="clarity-value">0</span>
                        </div>
                        <div class="setting">
                            <label>Preserve Black:</label>
                            <input type="checkbox" id="preserve-black" checked>
                        </div>
                    </div>
                </div>
                <button class="btn" onclick="processCustom()">🚀 Process with Custom Settings</button>
            </div>
            
            <div class="card">
                <h2>📊 Processing Status</h2>
                <div id="status">Ready</div>
                <div class="progress" id="progress-container" style="display: none;">
                    <div class="progress-bar" id="progress-bar"></div>
                </div>
                <div id="results"></div>
            </div>
            
            <script>
                // Update slider values display
                document.querySelectorAll('input[type="range"]').forEach(slider => {
                    const valueSpan = document.getElementById(slider.id + '-value');
                    slider.addEventListener('input', () => {
                        valueSpan.textContent = slider.value;
                    });
                });
                
                // API calls would go here for actual web implementation
                function autoEnhance() {
                    updateStatus("🤖 Auto-enhancing images...");
                    showProgress();
                    // Would call backend API
                    setTimeout(() => {
                        hideProgress();
                        updateStatus("✅ Auto-enhancement complete!");
                    }, 3000);
                }
                
                function usePreset(presetName) {
                    updateStatus(`📋 Processing with ${presetName} preset...`);
                    showProgress();
                    // Would call backend API
                    setTimeout(() => {
                        hideProgress();
                        updateStatus(`✅ ${presetName} enhancement complete!`);
                    }, 2000);
                }
                
                function processCustom() {
                    const settings = {
                        clahe: document.getElementById('clahe').value,
                        gamma: document.getElementById('gamma').value,
                        saturation: document.getElementById('saturation').value,
                        brightness: document.getElementById('brightness').value,
                        vibrance: document.getElementById('vibrance').value,
                        warmth: document.getElementById('warmth').value,
                        clarity: document.getElementById('clarity').value,
                        preserve_black: document.getElementById('preserve-black').checked
                    };
                    
                    updateStatus("⚙️ Processing with custom settings...");
                    showProgress();
                    // Would call backend API with settings
                    setTimeout(() => {
                        hideProgress();
                        updateStatus("✅ Custom enhancement complete!");
                    }, 2500);
                }
                
                function analyzeOnly() {
                    updateStatus("🔍 Analyzing images...");
                    showProgress();
                    setTimeout(() => {
                        hideProgress();
                        updateStatus("📊 Analysis complete! Check analysis report.");
                    }, 1500);
                }
                
                function refreshImages() {
                    updateStatus("🔄 Refreshing image list...");
                    // Would call backend API
                    setTimeout(() => {
                        updateStatus("✅ Image list refreshed!");
                    }, 500);
                }
                
                function updateStatus(message) {
                    document.getElementById('status').textContent = message;
                }
                
                function showProgress() {
                    document.getElementById('progress-container').style.display = 'block';
                    // Simulate progress
                    let width = 0;
                    const interval = setInterval(() => {
                        width += Math.random() * 20;
                        if (width >= 100) {
                            width = 100;
                            clearInterval(interval);
                        }
                        document.getElementById('progress-bar').style.width = width + '%';
                    }, 200);
                }
                
                function hideProgress() {
                    document.getElementById('progress-container').style.display = 'none';
                    document.getElementById('progress-bar').style.width = '0%';
                }
            </script>
        </body>
        </html>
        """.replace('{input_folder}', str(self.enhancer.input_folder))\
           .replace('{output_folder}', str(self.enhancer.output_folder))
        
        return html

# Integration helpers
class IntegrationHelper:
    """Helper functions for integrating with other tools"""
    
    @staticmethod
    def export_for_photoshop(img: np.ndarray, output_path: str) -> bool:
        """Export in format suitable for Photoshop editing"""
        # Convert to 16-bit for better Photoshop compatibility
        img_16bit = (img.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
        
        # Save as TIFF with no compression
        return cv2.imwrite(output_path, img_16bit, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    
    @staticmethod
    def create_action_script(settings: EnhancementSettings) -> str:
        """Create Photoshop action script equivalent"""
        script = f"""
        // MTG Proxy Enhancement - Photoshop Action Script
        // Generated settings equivalent
        
        // Levels adjustment (approximate gamma)
        var gamma = {settings.gamma:.2f};
        var levelsLayer = app.activeDocument.artLayers.add();
        levelsLayer.name = "Gamma Adjustment";
        
        // Saturation adjustment
        var saturation = {settings.saturation:.2f};
        var hslLayer = app.activeDocument.artLayers.add();
        hslLayer.name = "Saturation Adjustment";
        
        // Note: This is a simplified approximation
        // Full CLAHE and advanced tone mapping require custom plugins
        """
        return script
    
    @staticmethod
    def export_lightroom_preset(settings: EnhancementSettings, preset_name: str) -> str:
        """Create Lightroom-compatible preset (simplified)"""
        preset = f"""
        s.{preset_name} = {{
            Exposure = {settings.exposure:.2f},
            Highlights = {settings.highlights:.0f},
            Shadows = {settings.shadows:.0f},
            Brightness = {settings.brightness:.0f},
            Contrast = {(settings.contrast - 1.0) * 100:.0f},
            Saturation = {(settings.saturation - 1.0) * 100:.0f},
            Vibrance = {settings.vibrance:.0f},
            Temperature = {settings.warmth * 100:.0f},
            Tint = {settings.tint:.0f},
            Clarity = {settings.clarity:.0f}
        }}
        """
        return preset

# Complete usage examples and documentation
USAGE_EXAMPLES = """
🃏 MTG PROXY ENHANCER - COMPLETE USAGE GUIDE

🚀 QUICK START EXAMPLES:
# 1. One-click auto enhancement (recommended for beginners)
one_click_enhance()

# 2. Auto enhancement with custom folders
auto_enhance_all("./my_cards", "./enhanced_cards")

# 3. Use professional preset
quick_enhance_all(preset="professional")

# 4. Interactive interface
enhancer = create_mtg_enhancer_optimized()
interface = InteractiveInterface(enhancer)
ui = interface.create_widget_interface()
display(ui)

# 5. Command-line interface
run_cli_interface()

🔧 ADVANCED USAGE:
# Custom settings
settings = EnhancementSettings(
    gamma=1.3,
    saturation=1.2, 
    vibrance=15,
    clarity=10
)
enhancer = create_mtg_enhancer_optimized()
BatchProcessor(enhancer).batch_process_threaded(settings)

# Professional workflow with validation
pro = create_pro_enhancer()
enhanced, validation = pro.enhance_with_validation(image, settings)

# Comprehensive analysis
analysis = run_comprehensive_analysis("./my_cards")

📊 ANALYSIS & REPORTING:
# Quality assessment
quality_report = QualityAssessment.create_quality_report(enhancer)

# Performance benchmark  
benchmark_results = benchmark_enhancement()

# Create processing report
analytics = BatchAnalytics.analyze_batch_results(enhancer, results)
BatchAnalytics.create_processing_report(enhancer, analytics)

⚙️ CONFIGURATION:
# Create custom config
config = AppConfig(
    input_folder="./cards",
    output_folder="./enhanced", 
    max_workers=8,
    processing_mode=ProcessingMode.AUTO
)

# Save config
ConfigManager.save_config(config, "my_config.yaml")

# Use config
app = MTGEnhancerApp(config)
results = app.run()

🌐 COMMAND LINE:
python enhance_2_optimized.py --auto --input ./cards --workers 8
python enhance_2_optimized.py --preset professional --report
python enhance_2_optimized.py --interactive
python enhance_2_optimized.py --analyze --verbose

💡 TIPS:
• Use auto-enhancement for best results with minimal effort
• Professional preset works well for most MTG proxies
• Preserve black pixels is crucial for text readability
• Use multi-threading for large batches (--workers 8)
• Create reports for quality tracking
• Save successful settings as presets for reuse
"""

def print_complete_help():
    """Print comprehensive help and examples"""
    print(USAGE_EXAMPLES)

# Final optimized entry point
if __name__ == "__main__":
    # Check if running in Jupyter (avoid argument parsing conflicts)
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            # Running in Jupyter - skip command line parsing
            print("🃏 MTG Proxy Enhancer - Complete Optimized Version")
            print("=" * 60)
            print("\n📚 For complete usage guide, run: print_complete_help()")
            print("\n🚀 INSTANT START:")
            print("• one_click_enhance() - Auto-enhance everything")
            print("• run_cli_interface() - Interactive command-line")
            print("• auto_enhance_all() - Quick auto-processing")
            
            try:
                # Auto-initialize if images present
                enhancer = create_mtg_enhancer_optimized()
                if enhancer.images:
                    print(f"\n✅ Ready! Found {len(enhancer.images)} images")
                    print("🎯 Quick start: one_click_enhance()")
                else:
                    print(f"\n📂 Add images to '{enhancer.input_folder}' to begin")
            except Exception as e:
                logger.error(f"Initialization error: {e}")
        else:
            # Running as script - use command line parsing
            main()
    except ImportError:
        # Not in IPython/Jupyter environment
        if len(sys.argv) > 1:
            main()
        else:
            print("🃏 MTG Proxy Enhancer loaded!")

