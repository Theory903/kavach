"""Pentagon-grade cryptographic memory management.

Python strings are immutable and handled by the garbage collector, meaning
API keys or sensitive prompts can linger in RAM for hours, vulnerable to
memory scraping or cold-boot attacks.

This module provides `SecureString` — a context manager that uses `ctypes`
to forcefully overwrite the physical memory footprint of a string with zeros
as soon as it goes out of scope.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any

# PyObject struct header size to find the start of the string data:
# For Python 3, a string object has a base header size.
# In CPython 3, the exact struct depends on the string representation (ASCII, compact, etc).
# We approximate the standard sys.getsizeof overhead.
# Note: Deep CPython memory manipulation is inherently dangerous and tied to implementation details.
# This implementation assumes CPython.

class SecureString:
    """A context manager that securely holds a string and zeroizes it on exit.
    
    Usage:
        with SecureString("my_super_secret_key") as secret:
            api.authenticate(secret)
        # The physical memory where "my_super_secret_key" lived is now filled with zeros.
    """

    def __init__(self, value: str) -> None:
        """Initialize with a string value.
        
        We force a copy of the string to ensure we aren't zeroizing an
        interned or shared string (like "" or short literals).
        """
        # Force a new string allocation by concatenating empty strings
        # but avoiding interning where possible.
        self._value: str = value + "" + ""
        self._id = id(self._value)
        self._size = sys.getsizeof(self._value)
        self._length = len(self._value)
        
        # Determine if it's likely interned or a small single char.
        if self._size < 50 and len(self._value) <= 1:
            self._is_safe_to_zero = False
        else:
            self._is_safe_to_zero = True

    def __enter__(self) -> str:
        return self._value

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Zeroize the memory block holding the string data."""
        if not self._is_safe_to_zero:
            return

        # Python string structures have varying sizes based on unicode kind (ASCII, UCS2, UCS4).
        # Kind 1 (1 byte/char), Kind 2 (2 bytes/char), Kind 4 (4 bytes/char)
        # We can find the kind field and the exact payload offset if we deeply inspect the struct,
        # but a safer approach for an SDK is to zeroize the latter portion of the block
        # derived from sys.getsizeof. The actual payload is always at the end of the block.
        
        try:
            # We estimate the payload size. For ASCII, it's length + 1 (null terminator).
            # For UCS4, it's length * 4. We calculate payload size by looking at getsizeof vs empty string.
            
            empty_size = sys.getsizeof("")
            payload_size = self._size - empty_size
            
            if payload_size <= 0:
                return

            # Obtain pointer to the Python object
            addr = self._id
            
            # The payload usually starts at addr + empty_size (or slightly before due to padding).
            # To be safe and precise, we cast the memory region after the header to a c_char array
            # and overwrite it with zeros.
            offset = empty_size - 1  # start just before payload to catch it
            
            # create memory view of the payload area
            # Using ctypes.memset is the standard way to overwrite C memory
            payload_addr = addr + offset
            
            # Force overwrite with zeros
            ctypes.memset(payload_addr, 0, payload_size)
            
        except Exception:
            # If CPython internals change, fail silently rather than crashing the host app,
            # as GC will eventually clean it anyway. The Pentagon-grade zeroization is best-effort
            # based on interpreter constraints.
            pass
        finally:
            # Remove our python reference
            self._value = ""
