"""
POSIX-compliant cksum implementation

This module provides a replacement for the pycksum package that is compatible
with Python 3.12. It implements the same algorithm as the Unix cksum command.
"""

def cksum(file_obj_or_data):
    """
    Calculate POSIX cksum checksum.
    
    This function implements the same algorithm as the Unix cksum command,
    providing a drop-in replacement for pycksum.cksum that works with Python 3.12.
    
    Args:
        file_obj_or_data: Either a file object with a read() method, or bytes/string data
        
    Returns:
        int: The checksum as an unsigned 32-bit integer
        
    Examples:
        >>> with open('file.txt', 'rb') as f:
        ...     checksum = cksum(f)
        >>> checksum = cksum(b'hello world')
        >>> checksum = cksum('hello world')
    """
    if hasattr(file_obj_or_data, 'read'):
        # It's a file object
        data = file_obj_or_data.read()
        if hasattr(file_obj_or_data, 'seek'):
            file_obj_or_data.seek(0)  # Reset file pointer for potential reuse
    else:
        # It's raw data
        data = file_obj_or_data
    
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    # POSIX cksum algorithm implementation
    crc = 0
    length = len(data)
    
    # Process each byte in the data
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = (crc << 1) ^ 0x04C11DB7  # POSIX CRC-32 polynomial
            else:
                crc <<= 1
            crc &= 0xFFFFFFFF
    
    # Append the length as bytes to the CRC calculation
    temp_length = length
    while temp_length > 0:
        byte = temp_length & 0xFF
        temp_length >>= 8
        crc ^= byte << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = (crc << 1) ^ 0x04C11DB7
            else:
                crc <<= 1
            crc &= 0xFFFFFFFF
    
    # Return the inverted result as unsigned 32-bit integer
    return crc ^ 0xFFFFFFFF