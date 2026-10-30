import unittest

if __name__ == "__main__":
    result = unittest.defaultTestLoader.discover("tests")
    unittest.TextTestRunner(verbosity=2).run(result)
