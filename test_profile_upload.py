import unittest
import pandas as pd
import numpy as np
import io
from utils import process_uploaded_profile

class TestProfileUpload(unittest.TestCase):
    def test_process_uploaded_profile_solar(self):
        # Create a dummy CSV for Solar
        csv_content = "Hour,Solar_Output\n1,0.1\n2,0.5\n3,0.9"
        # We need to buffer it to mimic a file object
        file_obj = io.BytesIO(csv_content.encode('utf-8'))
        
        # Test with solar keywords
        profile = process_uploaded_profile(file_obj, keywords=['solar'])
        
        self.assertIsNotNone(profile)
        self.assertEqual(len(profile), 8760, "Should be padded to 8760")
        self.assertEqual(profile[0], 0.1)
        self.assertEqual(profile[1], 0.5)
        self.assertEqual(profile[2], 0.9)
        self.assertEqual(profile[3], 0.0, "Should be padded with 0")

    def test_process_uploaded_profile_fallback(self):
        # Create a dummy CSV with just numbers
        csv_content = "RandomCol\n0.5\n0.6"
        file_obj = io.BytesIO(csv_content.encode('utf-8'))
        
        # Test with no matching keywords (should fall back to numeric)
        profile = process_uploaded_profile(file_obj, keywords=['nothing'])
        
        self.assertIsNotNone(profile)
        self.assertEqual(profile[0], 0.5)

if __name__ == '__main__':
    unittest.main()
