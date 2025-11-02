#!/usr/bin/env python3
"""
Basic tests for the Object Detection app
"""

import unittest
import numpy as np
import cv2
import tempfile
import os

# Mock streamlit to prevent session state issues during testing
class MockStreamlit:
    def __init__(self):
        self.session_state = {}
        
    def set_page_config(self, *args, **kwargs):
        pass
        
    def title(self, *args, **kwargs):
        pass
        
    def header(self, *args, **kwargs):
        pass
        
    def columns(self, *args, **kwargs):
        return [MockColumn(), MockColumn()]
        
    def empty(self):
        return MockEmpty()
        
    def button(self, *args, **kwargs):
        return False
        
    def slider(self, *args, **kwargs):
        return 0.25
        
    def selectbox(self, *args, **kwargs):
        return "yolov8n.pt"
        
    def file_uploader(self, *args, **kwargs):
        return None
        
    def sidebar(self):
        return MockSidebar()
        
    def markdown(self, *args, **kwargs):
        pass
        
    def write(self, *args, **kwargs):
        pass
        
    def info(self, *args, **kwargs):
        pass
        
    def error(self, *args, **kwargs):
        pass
        
    def warning(self, *args, **kwargs):
        pass
        
    def success(self, *args, **kwargs):
        pass
        
    def spinner(self, *args, **kwargs):
        return MockSpinner()
        
    def progress(self, *args, **kwargs):
        return MockProgress()
        
    def cache_resource(self, func):
        return func
        
    def number_input(self, *args, **kwargs):
        return 20
        
    def dataframe(self, *args, **kwargs):
        pass
        
    def image(self, *args, **kwargs):
        pass
        
    def video(self, *args, **kwargs):
        pass
        
    def download_button(self, *args, **kwargs):
        pass

class MockColumn:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

class MockEmpty:
    def empty(self):
        return self
    def text(self, *args, **kwargs):
        pass
    def image(self, *args, **kwargs):
        pass
    def success(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass

class MockSpinner:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

class MockProgress:
    def progress(self, *args, **kwargs):
        return self
    def empty(self):
        pass

class MockSidebar:
    def header(self, *args, **kwargs):
        pass
    def write(self, *args, **kwargs):
        pass
    def file_uploader(self, *args, **kwargs):
        return None
    def number_input(self, *args, **kwargs):
        return 20
    def button(self, *args, **kwargs):
        return False

# Create a mock streamlit module
import sys
st = MockStreamlit()
sys.modules['streamlit'] = st

# Now we can safely import the functions from app.py
from app import draw_boxes, parse_results

class TestObjectDetection(unittest.TestCase):
    """Test cases for object detection functions"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a dummy image for testing
        self.dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create dummy class names
        self.dummy_class_names = {
            0: 'person',
            1: 'car',
            2: 'bicycle'
        }

    def test_draw_boxes_with_empty_boxes(self):
        """Test draw_boxes with no boxes"""
        result = draw_boxes(self.dummy_image, [], [], [], self.dummy_class_names)
        self.assertEqual(result.shape, self.dummy_image.shape)
        # Should return the same image when no boxes
        np.testing.assert_array_equal(result, self.dummy_image)

    def test_draw_boxes_with_valid_box(self):
        """Test draw_boxes with a valid bounding box"""
        boxes = [[10, 10, 50, 50]]
        scores = [0.8]
        class_ids = [0]
        
        result = draw_boxes(self.dummy_image, boxes, scores, class_ids, self.dummy_class_names)
        self.assertEqual(result.shape, self.dummy_image.shape)
        
        # The result should be different from the input due to the drawn box
        self.assertFalse(np.array_equal(result, self.dummy_image))

    def test_parse_results_empty(self):
        """Test parse_results with empty results"""
        boxes, scores, class_ids = parse_results([], None)
        self.assertEqual(boxes, [])
        self.assertEqual(scores, [])
        self.assertEqual(class_ids, [])

    def test_parse_results_with_no_boxes(self):
        """Test parse_results with results but no boxes"""
        # Create a mock results object with None boxes
        class MockResult:
            def __init__(self):
                self.boxes = None
        
        mock_results = [MockResult()]
        boxes, scores, class_ids = parse_results(mock_results, None)
        self.assertEqual(boxes, [])
        self.assertEqual(scores, [])
        self.assertEqual(class_ids, [])

    def test_draw_boxes_with_invalid_coordinates(self):
        """Test draw_boxes with invalid box coordinates"""
        boxes = [[10, 10, 5, 5]]  # x2 < x1, y2 < y1
        scores = [0.8]
        class_ids = [0]
        
        result = draw_boxes(self.dummy_image, boxes, scores, class_ids, self.dummy_class_names)
        # Should handle invalid coordinates gracefully
        self.assertEqual(result.shape, self.dummy_image.shape)

    def test_draw_boxes_with_out_of_bounds_coordinates(self):
        """Test draw_boxes with coordinates outside image bounds"""
        boxes = [[-10, -10, 150, 150]]  # Outside image bounds
        scores = [0.8]
        class_ids = [0]
        
        result = draw_boxes(self.dummy_image, boxes, scores, class_ids, self.dummy_class_names)
        # Should handle out of bounds coordinates gracefully
        self.assertEqual(result.shape, self.dummy_image.shape)

    def test_imports(self):
        """Test that all required modules can be imported"""
        try:
            import cv2
            import numpy as np
            import tempfile
            import os
            import yaml
            from pathlib import Path
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import required module: {e}")

if __name__ == '__main__':
    # Run the tests
    print("Running Object Detection App Tests...")
    unittest.main(verbosity=2)
