#!/usr/bin/env python3
"""Test configurator validation."""

import sys
sys.path.insert(0, '.')

from ueaj.utils.configurator import config, override


print("=== Test 1: Invalid parameter name ===")
@config
def func1(a: int, b: int = 10):
    return a + b

try:
    # This should fail - 'c' is not a valid parameter
    func1_bad = func1.override(c=20)
    print("❌ Should have failed!")
except TypeError as e:
    print(f"✅ Caught expected error: {e}")


print("\n=== Test 2: Valid override ===")
# This should work
func1_good = func1.override(b=20)
result = func1_good(5)
print(f"✅ Valid override worked: func1_good(5) = {result}")


print("\n=== Test 3: Function with **kwargs ===")
@config
def func2(a: int, **kwargs):
    return {"a": a, "kwargs": kwargs}

# This should work - func2 accepts **kwargs
func2_with_extra = func2.override(b=20, c=30)
result = func2_with_extra(5)
print(f"✅ Override with **kwargs worked: {result}")


print("\n=== Test 4: Nested function override validation ===")
@config
def helper(x: int, y: int = 1):
    return x * y

@config
def main_func(a: int, helper=helper):
    return helper(a)

try:
    # This should fail - 'z' is not a valid parameter for helper
    main_bad = main_func.override(
        helper=override(z=10)
    )
    # The error will happen when we call it and it tries to apply the override
    result = main_bad(5)
    print("❌ Should have failed!")
except TypeError as e:
    print(f"✅ Caught expected error: {e}")


print("\n=== Test 5: Multiple overrides with mixed validity ===")
@config
def func3(a: int, b: int = 10, c: str = "hello"):
    return f"{c}: {a + b}"

try:
    # Mix of valid and invalid
    func3_bad = func3.override(
        b=20,      # valid
        c="bye",   # valid  
        d=40       # invalid
    )
    print("❌ Should have failed!")
except TypeError as e:
    print(f"✅ Caught expected error: {e}")


print("\nAll tests passed!")