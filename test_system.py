import unittest
from app import app
import io

class TestSystem(unittest.TestCase):
    def setUp(self):
        # Set the app configuration to TESTING mode
        app.config['TESTING'] = True
        self.app = app.test_client()
    
    def test_full_system(self):
        # Use an image file for testing (ensure the path is valid)
        with open(r'C:\Users\jayve\OneDrive\Desktop\pest_detection_app\data\train\aphids\jpg_2.jpg', 'rb') as img:
            # Prepare the file upload request with multipart/form-data
            data = {
                'file': (img, 'aphid.jpg')
            }
            response = self.app.post('/', data=data, content_type='multipart/form-data')

            # Check if the request was successful
            self.assertEqual(response.status_code, 200, "Failed to load page or upload image")

            # Check if the pest name 'Aphid' is mentioned in the response data
            self.assertIn(b'Aphid', response.data, "Pest type 'Aphid' not found in the response")

            # Check if the control measures related to 'Aphid' are mentioned
            self.assertIn(b'Use insecticidal soap', response.data, "Control measure for 'Aphid' not found in the response")

if __name__ == '__main__':
    unittest.main()
