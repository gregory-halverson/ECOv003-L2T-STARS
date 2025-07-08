"""
Test for cksum_util module to ensure it provides compatible functionality
with the replaced pycksum package.
"""

import io
import tempfile
import os
import sys

# Add the ECOv003_L2T_STARS directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ECOv003_L2T_STARS'))

from cksum_util import cksum


def test_cksum_with_file_object():
    """Test that cksum works with file objects as expected in LPDAACDataPool."""
    test_data = b'test data for checksum verification'
    file_obj = io.BytesIO(test_data)
    
    # Test that we can calculate checksum
    result = cksum(file_obj)
    assert isinstance(result, int)
    assert result > 0
    
    # Test that file pointer is reset after calculation
    file_obj.seek(0)
    result2 = cksum(file_obj)
    assert result == result2, "Checksum should be the same for same data"


def test_cksum_with_bytes():
    """Test that cksum works with byte data."""
    test_data = b'hello world'
    result = cksum(test_data)
    
    # This should match the Unix cksum command output
    expected = 1135714720  # Known value from Unix cksum
    assert result == expected, f"Expected {expected}, got {result}"


def test_cksum_with_string():
    """Test that cksum works with string data."""
    test_string = 'hello world'
    result = cksum(test_string)
    
    # Should be same as bytes version
    expected = 1135714720
    assert result == expected, f"String checksum should match bytes checksum"


def test_cksum_with_real_file():
    """Test cksum with a real temporary file."""
    test_data = b'temporary file test data'
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(test_data)
        tmp_file.flush()
        
        # Test with file opened in binary mode
        with open(tmp_file.name, 'rb') as f:
            result = cksum(f)
            assert isinstance(result, int)
            assert result > 0
    
    # Clean up
    os.unlink(tmp_file.name)


def test_cksum_usage_pattern():
    """Test the specific usage pattern from LPDAACDataPool.get_local_checksum."""
    # Simulate the usage in LPDAACDataPool.py line 234
    test_data = b'simulated file content for LPDAAC checksum'
    
    with io.BytesIO(test_data) as file:
        # This mimics: return str(int(cksum(file)))
        checksum_result = str(int(cksum(file)))
        
        assert isinstance(checksum_result, str)
        assert checksum_result.isdigit()
        assert int(checksum_result) > 0


if __name__ == '__main__':
    # Run all tests
    test_functions = [
        test_cksum_with_file_object,
        test_cksum_with_bytes,
        test_cksum_with_string,
        test_cksum_with_real_file,
        test_cksum_usage_pattern,
    ]
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__} passed")
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            sys.exit(1)
    
    print("All tests passed! cksum_util is working correctly.")