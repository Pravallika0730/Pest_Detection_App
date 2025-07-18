import unittest
from app import app
import io

class TestIntegration(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()
    
    def test_upload_and_detection(self):
        # Use an image file for testing
        with open(r'C:\Users\jayve\OneDrive\Desktop\pest_detection_app\data\train\aphids\jpg_2.jpg', 'rb') as img:
            # Create a test file upload
            data = {
                'file': (img, 'aphid.jpg')
            }
            # Post the file to the '/' route
            response = self.app.post('/', data=data, content_type='multipart/form-data')

            # Ensure the request is successful
            self.assertEqual(response.status_code, 200)

            # Check if the pest type and control measures are in the response
            self.assertIn(b'Aphid', response.data)
            self.assertIn(b'Use insecticidal soap', response.data)

if __name__ == '__main__':
    unittest.main()
