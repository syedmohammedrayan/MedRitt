"""Backend regression tests."""

import os
os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-unit-tests')
from config import settings
