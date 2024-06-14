# tests/test_my_class.py

import unittest
from WokeyTalky.my_class import MyClass

class TestMyClass(unittest.TestCase):
    def test_some_method(self):
        obj = MyClass(1, 2)
        # Test the behavior of some_method()
        # Add assertions as needed
        pass

if __name__ == '__main__':
    unittest.main()