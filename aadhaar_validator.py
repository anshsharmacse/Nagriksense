# aadhaar_validator.py
import re
import requests
import hashlib
from datetime import datetime

class AadhaarValidator:
    def __init__(self):
        """Initialize Aadhaar validation system"""
        self.verhoeff_table_d = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
            [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
            [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
            [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
            [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
            [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
            [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
            [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        ]
        
        self.verhoeff_table_p = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
            [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
            [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
            [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
            [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
            [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
            [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
        ]
        
        self.verhoeff_table_inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
    
    def validate_format(self, aadhaar):
        """Validate Aadhaar number format"""
        # Remove spaces and hyphens
        aadhaar = re.sub(r'[\s\-]', '', str(aadhaar))
        
        # Check if it's exactly 12 digits
        if len(aadhaar) != 12:
            return False, "Aadhaar number must be exactly 12 digits"
        
        # Check if all characters are digits
        if not aadhaar.isdigit():
            return False, "Aadhaar number must contain only digits"
        
        # Check if it starts with 0 or 1 (invalid for Aadhaar)
        if aadhaar[0] in ['0', '1']:
            return False, "Aadhaar number cannot start with 0 or 1"
        
        return True, aadhaar
    
    def verhoeff_checksum(self, aadhaar):
        """Validate Aadhaar using Verhoeff algorithm"""
        aadhaar_digits = [int(d) for d in reversed(aadhaar)]
        c = 0
        
        for i, digit in enumerate(aadhaar_digits):
            c = self.verhoeff_table_d[c][self.verhoeff_table_p[i % 8][digit]]
        
        return c == 0
    
    def validate_aadhaar(self, aadhaar):
        """Complete Aadhaar validation"""
        # Format validation
        is_valid_format, result = self.validate_format(aadhaar)
        if not is_valid_format:
            return False, result
        
        aadhaar = result
        
        # Verhoeff algorithm validation
        if not self.verhoeff_checksum(aadhaar):
            return False, "Invalid Aadhaar number (checksum failed)"
        
        return True, "Valid Aadhaar number"
    
    def simulate_uidai_verification(self, aadhaar, consent=True):
        """Simulate UIDAI API verification (for demo purposes)"""
        if not consent:
            return False, "User consent required for Aadhaar verification"
        
        # Simulate API call delay
        import time
        time.sleep(1)
        
        # Basic validation
        is_valid, message = self.validate_aadhaar(aadhaar)
        if not is_valid:
            return False, message
        
        # Simulate realistic API response
        verification_result = {
            'status': 'success',
            'aadhaar_verified': True,
            'timestamp': datetime.now().isoformat(),
            'reference_id': hashlib.md5(aadhaar.encode()).hexdigest()[:8],
            'message': 'Aadhaar number verified successfully'
        }
        
        return True, verification_result
    
    def mask_aadhaar(self, aadhaar):
        """Mask Aadhaar number for privacy (show only last 4 digits)"""
        if len(aadhaar) == 12:
            return 'XXXX-XXXX-' + aadhaar[-4:]
        return 'XXXX-XXXX-XXXX'
    
    def generate_test_aadhaar(self):
        """Generate valid test Aadhaar numbers for development"""
        # This is for testing purposes only
        base = "234567890"  # 9 digits
        
        # Calculate checksum using Verhoeff algorithm
        digits = [int(d) for d in reversed(base + "00")]  # Add temporary checksum
        c = 0
        
        for i, digit in enumerate(digits):
            c = self.verhoeff_table_d[c][self.verhoeff_table_p[i % 8][digit]]
        
        checksum = self.verhoeff_table_inv[c]
        return base + f"{checksum:02d}"

class AadhaarCompliance:
    """Ensure compliance with Aadhaar Act 2016 and data protection laws"""
    
    @staticmethod
    def get_consent_text():
        """Get legally compliant consent text"""
        return """
        CONSENT FOR AADHAAR VERIFICATION
        
        I hereby give my consent for verification of my Aadhaar number for the purpose of:
        - Identity verification for civic enforcement system
        - Maintaining records as per legal requirements
        - Sending notifications regarding civic violations
        
        I understand that:
        - My Aadhaar information will be used only for stated purposes
        - Data will be stored securely and not shared with unauthorized parties
        - I have the right to withdraw consent and request data deletion
        - This consent is voluntary and given under Aadhaar Act 2016
        
        By clicking 'I Agree', I provide my explicit consent for the above purposes.
        """
    
    @staticmethod
    def log_consent(aadhaar, user_ip, consent_given=True):
        """Log user consent for audit purposes"""
        consent_record = {
            'aadhaar_hash': hashlib.sha256(aadhaar.encode()).hexdigest(),
            'user_ip': user_ip,
            'consent_given': consent_given,
            'timestamp': datetime.now().isoformat(),
            'purpose': 'civic_enforcement_verification'
        }
        
        # In production, this would be stored in a secure audit log
        print(f"Consent logged: {consent_record}")
        return consent_record

if __name__ == "__main__":
    # Test Aadhaar validation
    validator = AadhaarValidator()
    
    # Test with generated valid Aadhaar
    test_aadhaar = validator.generate_test_aadhaar()
    print(f"Generated test Aadhaar: {test_aadhaar}")
    
    is_valid, message = validator.validate_aadhaar(test_aadhaar)
    print(f"Validation result: {is_valid}, {message}")
    
    # Test UIDAI simulation
    success, result = validator.simulate_uidai_verification(test_aadhaar, consent=True)
    print(f"UIDAI verification: {success}")
    if success:
        print(f"Result: {result}")
    
    # Test masking
    masked = validator.mask_aadhaar(test_aadhaar)
    print(f"Masked Aadhaar: {masked}")
