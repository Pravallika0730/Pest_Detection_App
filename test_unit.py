import unittest
import os
from app import identify_pest, get_control_measures

class TestPestDetection(unittest.TestCase):
    def test_identify_pest(self):
        # Correct path to your image
        test_image_path = r'C:\Users\jayve\OneDrive\Desktop\pest_detection_app\data\train\aphids\jpg_2.jpg'
        pest_type, particle_count = identify_pest(test_image_path)
        
        # Test if the function correctly identifies the pest type
        self.assertEqual(pest_type, 'Aphid')
        
        # Check if particle count is a positive float
        self.assertTrue(isinstance(particle_count, float))
        self.assertGreater(particle_count, 0)

if __name__ == '__main__':
    unittest.main()
