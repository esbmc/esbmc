# __ESBMC_old() Test Suite Summary

## Overview
Test suite for `__ESBMC_old()` functionality in ESBMC function contracts.
The `__ESBMC_old(expr)` captures the pre-state value of an expression for use in postconditions.

## Test Results (All Passed ✅)

### Basic Functionality Tests
| Test | Function | Description | Result |
|------|----------|-------------|--------|
| test_old_basic | increment_global | Basic integer global variable | ✅ PASS |
| test_old_bool | toggle_flag | Boolean type support | ✅ PASS |
| test_old_struct | move_point | Struct field access | ✅ PASS |
| test_old_multiple | swap | Multiple old() in same function | ✅ PASS |
| test_old_local | increment_param | Pointer parameter modification | ✅ PASS |

### Advanced Tests
| Test | Function | Description | Result |
|------|----------|-------------|--------|
| test_old_return | increment_and_return | old() with __ESBMC_return_value | ✅ PASS |
| test_old_complex | compute | Complex expression: old(x + y) | ✅ PASS |
| test_old_negative | withdraw | Subtraction and preconditions | ✅ PASS |

### Failure Detection Tests (Should FAIL)
| Test | Function | Description | Result |
|------|----------|-------------|--------|
| test_old_fail | increment_global_wrong | Detects x+1 instead of x | ✅ FAIL (correct) |
| test_old_wrong_impl | withdraw_wrong | Detects amount*2 instead of amount | ✅ FAIL (correct) |

## Usage Example

```c
int global = 0;

void increment_global(int x)
{
  __ESBMC_requires(x > 0);
  __ESBMC_ensures(global == __ESBMC_old(global) + x);

  global += x;
}
```

## Key Features Verified

1. ✅ **Type polymorphism**: Works with int, bool, struct fields, pointers
2. ✅ **Complex expressions**: Supports `old(x + y)`, `old(p.x)`, `old(*ptr)`
3. ✅ **Multiple snapshots**: Multiple `old()` in same ensures clause
4. ✅ **Integration**: Works with `__ESBMC_return_value`
5. ✅ **Error detection**: Correctly detects contract violations

## Implementation Notes

- Single declaration: `int __ESBMC_old(int);`
- Type inference at IR level via `arguments[0].type()`
- Automatic snapshot creation in wrapper function
- Contract cleanup in renamed original function

## Verification Command

```bash
./build/src/esbmc/esbmc --enforce-contract <function_name> <file.c>
```
