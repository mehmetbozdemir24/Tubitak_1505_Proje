"""
tests/conftest.py — pytest yapılandırması.

Kök dizindeki modüllerin (abac, tenancy, auth, api_schemas, ...) test
dosyalarından import edilebilmesi için proje kökünü sys.path'e ekler.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))