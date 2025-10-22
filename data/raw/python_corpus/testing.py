import sys
import traceback
from typing import Callable, Any, List, Dict, Optional
from functools import wraps

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.failures = []
    
    def add_success(self):
        self.passed += 1
    
    def add_failure(self, test_name: str, error: Exception):
        self.failed += 1
        self.failures.append((test_name, error))
    
    def add_error(self, test_name: str, error: Exception):
        self.failed += 1
        self.errors.append((test_name, error))

class AssertionError(Exception):
    pass

def assert_equal(actual: Any, expected: Any, msg: str = None):
    if actual != expected:
        message = msg or f"Expected {expected}, got {actual}"
        raise AssertionError(message)

def assert_true(condition: bool, msg: str = None):
    if not condition:
        message = msg or "Expected True, got False"
        raise AssertionError(message)

def assert_false(condition: bool, msg: str = None):
    if condition:
        message = msg or "Expected False, got True"
        raise AssertionError(message)

def assert_in(item: Any, container: Any, msg: str = None):
    if item not in container:
        message = msg or f"Expected {item} to be in {container}"
        raise AssertionError(message)

def assert_raises(exception_class: type):
    class ContextManager:
        def __init__(self, exc_class):
            self.exc_class = exc_class
            self.exception = None
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.exc_class.__name__} to be raised")
            if not issubclass(exc_type, self.exc_class):
                return False
            self.exception = exc_val
            return True
    
    return ContextManager(exception_class)

class TestRunner:
    def __init__(self):
        self.tests = []
        self.fixtures = {}
    
    def collect_tests(self, module):
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and name.startswith('test_'):
                self.tests.append((name, obj))
    
    def run_tests(self) -> TestResult:
        result = TestResult()
        
        for test_name, test_func in self.tests:
            try:
                print(f"Running {test_name}...", end=" ")
                test_func()
                result.add_success()
                print("PASSED")
            except AssertionError as e:
                result.add_failure(test_name, e)
                print("FAILED")
                print(f"  {e}")
            except Exception as e:
                result.add_error(test_name, e)
                print("ERROR")
                print(f"  {e}")
                traceback.print_exc()
        
        return result
    
    def print_summary(self, result: TestResult):
        total = result.passed + result.failed
        print(f"\n{'='*50}")
        print(f"Tests run: {total}")
        print(f"Passed: {result.passed}")
        print(f"Failed: {result.failed}")
        
        if result.failures:
            print("\nFailures:")
            for test_name, error in result.failures:
                print(f"  {test_name}: {error}")
        
        if result.errors:
            print("\nErrors:")
            for test_name, error in result.errors:
                print(f"  {test_name}: {error}")

def fixture(scope: str = "function"):
    def decorator(func: Callable):
        func._pytest_fixture = True
        func._pytest_scope = scope
        return func
    return decorator

def parametrize(argnames: str, argvalues: List[tuple]):
    def decorator(func: Callable):
        func._pytest_parametrize = True
        func._pytest_argnames = argnames.split(',')
        func._pytest_argvalues = argvalues
        return func
    return decorator

class TestClass:
    def setup_method(self):
        pass
    
    def teardown_method(self):
        pass

def main():
    runner = TestRunner()
    
    # Collect tests from current module
    current_module = sys.modules[__name__]
    runner.collect_tests(current_module)
    
    # Run tests
    result = runner.run_tests()
    runner.print_summary(result)
    
    return result.failed == 0

# Example tests
def test_addition():
    assert_equal(2 + 2, 4)
    assert_equal(1 + 1, 2)

def test_string_operations():
    assert_equal("hello".upper(), "HELLO")
    assert_in("world", "hello world")

def test_list_operations():
    lst = [1, 2, 3]
    assert_equal(len(lst), 3)
    assert_in(2, lst)

def test_exception_handling():
    with assert_raises(ValueError):
        int("not a number")

class TestMath:
    def setup_method(self):
        self.numbers = [1, 2, 3, 4, 5]
    
    def test_sum(self):
        assert_equal(sum(self.numbers), 15)
    
    def test_max(self):
        assert_equal(max(self.numbers), 5)

if __name__ == "__main__":
    main()
