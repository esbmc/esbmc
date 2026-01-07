# __ESBMC_old() User Guide

## Introduction

`__ESBMC_old(expr)` captures the pre-state value of an expression for use in function postconditions. This is essential for specifying how function execution changes program state.

## Syntax

```c
void function_name(parameters)
{
  __ESBMC_requires(precondition);
  __ESBMC_ensures(postcondition_with_old);

  // function body
}
```

## Basic Example

```c
int counter = 0;

void increment(int delta)
{
  __ESBMC_requires(delta > 0);
  __ESBMC_ensures(counter == __ESBMC_old(counter) + delta);

  counter += delta;
}
```

## Common Patterns

### 1. Simple State Change
```c
__ESBMC_ensures(x == __ESBMC_old(x) + 1);
```

### 2. With Precondition
```c
__ESBMC_requires(__ESBMC_old(balance) >= amount);
__ESBMC_ensures(balance == __ESBMC_old(balance) - amount);
```

### 3. Multiple Variables
```c
__ESBMC_ensures(x == __ESBMC_old(y));
__ESBMC_ensures(y == __ESBMC_old(x));
```

### 4. Struct Fields
```c
__ESBMC_ensures(point.x == __ESBMC_old(point.x) + dx);
```

### 5. Complex Expressions
```c
__ESBMC_ensures(x + y == __ESBMC_old(x + y) + delta);
```

### 6. With Return Value
```c
__ESBMC_ensures(__ESBMC_return_value == __ESBMC_old(counter) + 1);
```

## Type Support

Works with all C types:
- Integer types: `int`, `long`, `unsigned`, etc.
- Boolean: `_Bool` / `bool`
- Pointers: `int*`, `void*`, etc.
- Struct fields: `struct.field`
- Array elements: `arr[i]`
- Complex expressions: `x + y`, `*ptr`, etc.

## Verification Command

```bash
# Verify function satisfies its contract
esbmc --enforce-contract function_name file.c

# Replace calls with contracts
esbmc --replace-call-with-contract function_name file.c
```

## Common Mistakes

❌ **Don't** use in requires (old is implicit in requires):
```c
__ESBMC_requires(__ESBMC_old(x) > 0);  // Wrong!
__ESBMC_requires(x > 0);               // Correct
```

✅ **Do** use in ensures to compare before/after:
```c
__ESBMC_ensures(x == __ESBMC_old(x) + 1);  // Correct
```

## Implementation Details

- Declaration: `int __ESBMC_old(int)`
- Type inference happens at IR conversion
- Creates snapshot before function call
- Replaces old() with snapshot in ensures clause

## More Examples

See test files in `regression/function_contract/test_old_*/`
