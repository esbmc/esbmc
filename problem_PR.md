
## Current Implementation

Added pre-scan for forward references (`-> 'Bar'`):
- Detects forward-referenced classes in method return types
- Recursively processes referenced class
- **Only registers class tag, NOT attributes**

**Test results:**
- ✅ `github_2997-1`: Empty `__init__`, no attributes  
- ❌ `github_2997-2`: Implicit constructor - KNOWNBUG
- ❌ `github_2997`: Has attributes (`self.x`) - KNOWNBUG

---

## Problem: Attribute Registration Fails
```python
class Foo:
    def __init__(self, x: int) -> None:
        self.x = x

    def bar(self) -> 'Bar':
        return Bar(self)

class Bar:
    def __init__(self, f: Foo) -> None:
        self.x = f.x

f: Foo = Foo(5)
b: Bar = f.bar()
assert b.x == 5
```
`ERROR: Type undefined for "x"`

**Why this fails:**
- get attributes  depends on current class name
- Symbol IDs generated as `Foo::Bar::x` instead of `Bar::x`

---

## Current idea: Second Scan with Scope Isolation

```cpp
// During pre-scan:

// 1. Save scope
std::string saved_class = current_class_name_;

// 2. Switch to Bar's scope
current_class_name_ = "Bar";

// 3. Fully process Bar (including attributes)
get_class_definition(bar_node, target_block);

// 4. Restore scope
current_class_name_ = saved_class;
```

---

## Problem 
Python frontend design assumes that class definitions are processed sequentially, relying on global state variables to track the current scope. Pre-scanning breaks this assumption but does not provide complete scope stack management.

**This PR documents the problem. Do NOT merge until scope isolation is implemented.**
