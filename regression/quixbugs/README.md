## This document contains the current ESBMC output for test cases that exhibit incorrect behavior or trigger segmentation faults during verification.

### breadth_first_search/main.py
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:4: error: Function is missing a type annotation  [no-untyped-def]
main.py:5: error: Need type annotation for "queue"  [var-annotated]
Found 2 errors in 1 file (checked 1 source file)

ERROR: Function "Queue" not found (main.py line 5)
```

### bucketsort/main.py
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:26: error: Call to untyped function "bucketsort" in typed context  [no-untyped-call]
main.py:27: error: Call to untyped function "bucketsort" in typed context  [no-untyped-call]
Found 3 errors in 1 file (checked 1 source file)

Converting
terminate called after throwing an instance of 'std::invalid_argument'
  what():  stoi
[1]    132750 IOT instruction (core dumped)  esbmc main.py
```

### depth_first_search/main.py
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:5: error: Function is missing a type annotation  [no-untyped-def]
main.py:13: error: Call to untyped function "search_from" in typed context  [no-untyped-call]
main.py:16: error: Call to untyped function "search_from" in typed context  [no-untyped-call]
Found 4 errors in 1 file (checked 1 source file)

ERROR: Function "search_from" not found (main.py line 16)
```

### detect_cycle/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
node.py:2: error: Function is missing a type annotation  [no-untyped-def]
node.py:4: error: Cannot assign to a method  [method-assign]
node.py:5: error: Cannot assign to a method  [method-assign]
node.py:6: error: Cannot assign to a method  [method-assign]
node.py:10: error: Function is missing a return type annotation  [no-untyped-def]
node.py:13: error: Function is missing a return type annotation  [no-untyped-def]
node.py:16: error: Function is missing a return type annotation  [no-untyped-def]
main.py:4: error: Function is missing a type annotation  [no-untyped-def]
main.py:33: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:34: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:35: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:36: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:37: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:40: error: Function is missing a return type annotation  [no-untyped-def]
main.py:40: note: Use "-> None" if function does not return a value
main.py:45: error: Call to untyped function "detect_cycle" in typed context  [no-untyped-call]
main.py:50: error: Function is missing a return type annotation  [no-untyped-def]
main.py:50: note: Use "-> None" if function does not return a value
main.py:55: error: Cannot assign to a method  [method-assign]
main.py:55: error: Incompatible types in assignment (expression has type "Node", variable has type "Callable[[], Any]")  [assignment]
main.py:57: error: Call to untyped function "detect_cycle" in typed context  [no-untyped-call]
main.py:62: error: Function is missing a return type annotation  [no-untyped-def]
main.py:62: note: Use "-> None" if function does not return a value
main.py:67: error: Cannot assign to a method  [method-assign]
main.py:67: error: Incompatible types in assignment (expression has type "Node", variable has type "Callable[[], Any]")  [assignment]
main.py:69: error: Call to untyped function "detect_cycle" in typed context  [no-untyped-call]
main.py:74: error: Function is missing a return type annotation  [no-untyped-def]
main.py:74: note: Use "-> None" if function does not return a value
main.py:79: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:80: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:82: error: Call to untyped function "detect_cycle" in typed context  [no-untyped-call]
main.py:87: error: Function is missing a return type annotation  [no-untyped-def]
main.py:87: note: Use "-> None" if function does not return a value
main.py:92: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:93: error: Call to untyped function "detect_cycle" in typed context  [no-untyped-call]
main.py:98: error: Function is missing a return type annotation  [no-untyped-def]
main.py:98: note: Use "-> None" if function does not return a value
main.py:103: error: Cannot assign to a method  [method-assign]
main.py:103: error: Incompatible types in assignment (expression has type "Node", variable has type "Callable[[], Any]")  [assignment]
main.py:105: error: Call to untyped function "detect_cycle" in typed context  [no-untyped-call]
main.py:109: error: Call to untyped function "test1" in typed context  [no-untyped-call]
main.py:110: error: Call to untyped function "test2" in typed context  [no-untyped-call]
main.py:111: error: Call to untyped function "test3" in typed context  [no-untyped-call]
main.py:112: error: Call to untyped function "test4" in typed context  [no-untyped-call]
main.py:113: error: Call to untyped function "test5" in typed context  [no-untyped-call]
main.py:114: error: Call to untyped function "test6" in typed context  [no-untyped-call]
Found 40 errors in 2 files (checked 1 source file)

Converting
ERROR: All parameters in function "Node" must be type annotated
```

### find_first_in_sorted/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:59: error: Call to untyped function "find_first_in_sorted" in typed context  [no-untyped-call]
main.py:60: error: Call to untyped function "find_first_in_sorted" in typed context  [no-untyped-call]
main.py:61: error: Call to untyped function "find_first_in_sorted" in typed context  [no-untyped-call]
main.py:62: error: Call to untyped function "find_first_in_sorted" in typed context  [no-untyped-call]
main.py:63: error: Call to untyped function "find_first_in_sorted" in typed context  [no-untyped-call]
main.py:64: error: Call to untyped function "find_first_in_sorted" in typed context  [no-untyped-call]
main.py:65: error: Call to untyped function "find_first_in_sorted" in typed context  [no-untyped-call]
Found 8 errors in 1 file (checked 1 source file)

ERROR: Type inference failed for Assign at line 7
```

### find_in_sorted/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:3: error: Function is missing a type annotation  [no-untyped-def]
main.py:8: error: Call to untyped function "binsearch" in typed context  [no-untyped-call]
main.py:10: error: Call to untyped function "binsearch" in typed context  [no-untyped-call]
main.py:14: error: Call to untyped function "binsearch" in typed context  [no-untyped-call]
Found 5 errors in 1 file (checked 1 source file)

ERROR: Function "binsearch" not found (main.py line 14)
```

### flatten/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:5: error: Call to untyped function "flatten" in typed context  [no-untyped-call]
main.py:11: error: Call to untyped function "flatten" in typed context  [no-untyped-call]
main.py:12: error: Call to untyped function "flatten" in typed context  [no-untyped-call]
main.py:13: error: Call to untyped function "flatten" in typed context  [no-untyped-call]
main.py:14: error: Call to untyped function "flatten" in typed context  [no-untyped-call]
main.py:15: error: Call to untyped function "flatten" in typed context  [no-untyped-call]
Found 7 errors in 1 file (checked 1 source file)

WARNING: Empty or malformed list literal detected. Using 'list[int]' as default (main.py:0)
WARNING: Mixed types detected in list literal: int vs list[int]. Using 'list[int]' as fallback (main.py:0)
WARNING: Mixed types detected in list literal: list[int] vs list[list[int]]. Using 'list[int]' as fallback (main.py:0)
WARNING: Mixed types detected in list literal: int vs list[list[int]]. Using 'list[int]' as fallback (main.py:0)
WARNING: Type inference conflict for parameter 0: list[list[int]] vs list[int]. Using 'int' as fallback (main.py:0)
Converting
ERROR: Invalid list access: could not resolve position or element type
```

### gcd/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:1: error: Function is missing a type annotation  [no-untyped-def]
main.py:5: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
main.py:28: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
main.py:29: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
main.py:30: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
main.py:31: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
main.py:32: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
main.py:33: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
Found 8 errors in 1 file (checked 1 source file)

Converting
Generating GOTO Program
WARNING: function should not return value

* file: main.py
* line: 3
* function: gcd
* column: 8
GOTO program creation time: 1.181s
GOTO program processing time: 0.010s
Starting Bounded Model Checking
[1]    79225 segmentation fault (core dumped)  esbmc main.py
```

### gcd_fail/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:1: error: Function is missing a type annotation  [no-untyped-def]
main.py:5: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
main.py:28: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
main.py:29: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
main.py:30: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
main.py:31: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
main.py:32: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
main.py:33: error: Call to untyped function "gcd" in typed context  [no-untyped-call]
Found 8 errors in 1 file (checked 1 source file)

Converting
Generating GOTO Program
WARNING: function should not return value

* file: main.py
* line: 3
* function: gcd
* column: 8
GOTO program creation time: 1.167s
GOTO program processing time: 0.010s
Starting Bounded Model Checking
[1]    82415 segmentation fault (core dumped)  esbmc main.py
```

### get_factors/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:8: error: Call to untyped function "get_factors" in typed context  [no-untyped-call]
main.py:24: error: Call to untyped function "get_factors" in typed context  [no-untyped-call]
main.py:25: error: Call to untyped function "get_factors" in typed context  [no-untyped-call]
main.py:26: error: Call to untyped function "get_factors" in typed context  [no-untyped-call]
main.py:27: error: Call to untyped function "get_factors" in typed context  [no-untyped-call]
main.py:28: error: Call to untyped function "get_factors" in typed context  [no-untyped-call]
main.py:29: error: Call to untyped function "get_factors" in typed context  [no-untyped-call]
main.py:30: error: Call to untyped function "get_factors" in typed context  [no-untyped-call]
main.py:31: error: Call to untyped function "get_factors" in typed context  [no-untyped-call]
main.py:32: error: Call to untyped function "get_factors" in typed context  [no-untyped-call]
main.py:33: error: Call to untyped function "get_factors" in typed context  [no-untyped-call]
main.py:34: error: Call to untyped function "get_factors" in typed context  [no-untyped-call]
Found 13 errors in 1 file (checked 1 source file)

WARNING: Type inference conflict for parameter 0: int vs list[int]. Using 'int' as fallback (main.py:1)
Converting
esbmc: esbmc/src/python-frontend/python_list.cpp:961: exprt python_list::compare(const exprt&, const exprt&, const string&): Assertion `lhs_symbol' failed.
[1]    20782 IOT instruction (core dumped)  esbmc main.py
```

### hanoi/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:6: error: Call to untyped function "hanoi" in typed context  [no-untyped-call]
main.py:8: error: Call to untyped function "hanoi" in typed context  [no-untyped-call]
main.py:12: error: Call to untyped function "hanoi" in typed context  [no-untyped-call]
main.py:13: error: Call to untyped function "hanoi" in typed context  [no-untyped-call]
main.py:14: error: Call to untyped function "hanoi" in typed context  [no-untyped-call]
main.py:15: error: Call to untyped function "hanoi" in typed context  [no-untyped-call]
main.py:16: error: Call to untyped function "hanoi" in typed context  [no-untyped-call]
main.py:17: error: Call to untyped function "hanoi" in typed context  [no-untyped-call]
main.py:18: error: Call to untyped function "hanoi" in typed context  [no-untyped-call]
main.py:19: error: Call to untyped function "hanoi" in typed context  [no-untyped-call]
Found 11 errors in 1 file (checked 1 source file)

ERROR: Object "" not found.
```

### is_valid_parenthesization/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:116: error: Call to untyped function "is_valid_parenthesization" in typed context  [no-untyped-call]
main.py:117: error: Call to untyped function "is_valid_parenthesization" in typed context  [no-untyped-call]
main.py:118: error: Call to untyped function "is_valid_parenthesization" in typed context  [no-untyped-call]
Found 4 errors in 1 file (checked 1 source file)

Converting
Generating GOTO Program
GOTO program creation time: 0.710s
GOTO program processing time: 0.010s
Checking base case, k = 1
Starting Bounded Model Checking
WARNING: no body for function __ESBMC_get_object_size
Not unwinding loop 3 iteration 1   file /home/bruno/projects/esbmc/src/c2goto/library/string.c line 110 column 12 function strcmp
Not unwinding loop 42 iteration 1   file main.py line 4 column 4 function is_valid_parenthesization
WARNING: no body for function __ESBMC_get_object_size
Not unwinding loop 3 iteration 1   file /home/bruno/projects/esbmc/src/c2goto/library/string.c line 110 column 12 function strcmp
Not unwinding loop 42 iteration 1   file main.py line 4 column 4 function is_valid_parenthesization
WARNING: no body for function __ESBMC_get_object_size
Not unwinding loop 3 iteration 1   file /home/bruno/projects/esbmc/src/c2goto/library/string.c line 110 column 12 function strcmp
Not unwinding loop 42 iteration 1   file main.py line 4 column 4 function is_valid_parenthesization
Symex completed in: 0.017s (132 assignments)
Caching time: 0.000s (removed 0 assertions)
Slicing time: 0.002s (removed 77 assignments)
Generated 22 VCC(s), 10 remaining after simplification (55 assignments)
No solver specified; defaulting to Boolector
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.004s
Solving with solver Boolector 3.2.4
Runtime decision procedure: 0.073s
Building error trace

[Counterexample]


State 1  thread 0
----------------------------------------------------
  ESBMC_length_0 = 0 (00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000)

State 3  thread 0
----------------------------------------------------
  ESBMC_length_0 = 4611686018427387904 (01000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000)

State 4 file main.py line 4 column 4 function is_valid_parenthesization thread 0
----------------------------------------------------
Violated property:
  file main.py line 4 column 4 function is_valid_parenthesization
  dereference failure: array bounds violated


VERIFICATION FAILED

Bug found (k = 1)
```

### kheapsort/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:15: error: Call to untyped function "kheapsort" in typed context  [no-untyped-call]
main.py:16: error: Call to untyped function "kheapsort" in typed context  [no-untyped-call]
main.py:17: error: Call to untyped function "kheapsort" in typed context  [no-untyped-call]
main.py:18: error: Call to untyped function "kheapsort" in typed context  [no-untyped-call]
Found 5 errors in 1 file (checked 1 source file)

Converting
ERROR: List slicing with missing bounds not yet supported
```

### knapsack/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:4: error: Need type annotation for "memo"  [var-annotated]
main.py:21: error: Call to untyped function "knapsack" in typed context  [no-untyped-call]
main.py:22: error: Call to untyped function "knapsack" in typed context  [no-untyped-call]
main.py:23: error: Call to untyped function "knapsack" in typed context  [no-untyped-call]
main.py:28: error: Call to untyped function "knapsack" in typed context  [no-untyped-call]
main.py:29: error: Call to untyped function "knapsack" in typed context  [no-untyped-call]
main.py:30: error: Call to untyped function "knapsack" in typed context  [no-untyped-call]
main.py:31: error: Call to untyped function "knapsack" in typed context  [no-untyped-call]
main.py:35: error: Call to untyped function "knapsack" in typed context  [no-untyped-call]
main.py:39: error: Call to untyped function "knapsack" in typed context  [no-untyped-call]
Found 11 errors in 1 file (checked 1 source file)

ERROR: Function "defaultdict" not found (main.py line 4)
```

### kth/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:11: error: Call to untyped function "kth" in typed context  [no-untyped-call]
main.py:13: error: Call to untyped function "kth" in typed context  [no-untyped-call]
main.py:17: error: Call to untyped function "kth" in typed context  [no-untyped-call]
main.py:18: error: Call to untyped function "kth" in typed context  [no-untyped-call]
main.py:19: error: Call to untyped function "kth" in typed context  [no-untyped-call]
main.py:20: error: Call to untyped function "kth" in typed context  [no-untyped-call]
main.py:21: error: Call to untyped function "kth" in typed context  [no-untyped-call]
main.py:22: error: Call to untyped function "kth" in typed context  [no-untyped-call]
main.py:23: error: Call to untyped function "kth" in typed context  [no-untyped-call]
Found 10 errors in 1 file (checked 1 source file)

Converting
Generating GOTO Program
WARNING: function should not return value

* file: main.py
* line: 15
* function: kth
* column: 8
GOTO program creation time: 0.789s
GOTO program processing time: 0.013s
Checking base case, k = 1
Starting Bounded Model Checking
Not unwinding loop 25 iteration 1   file /home/bruno/projects/esbmc/src/c2goto/library/string.c line 278 column 3 function __memcpy_impl
Symex completed in: 0.020s (257 assignments)
Caching time: 0.000s (removed 2 assertions)
Slicing time: 0.003s (removed 244 assignments)
Generated 17 VCC(s), 4 remaining after simplification (11 assignments)
No solver specified; defaulting to Boolector
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.004s
Solving with solver Boolector 3.2.4
Runtime decision procedure: 0.000s
BMC program time: 0.028s
No bug has been found in the base case
Checking forward condition, k = 1
Starting Bounded Model Checking
Not unwinding loop 25 iteration 1   file /home/bruno/projects/esbmc/src/c2goto/library/string.c line 278 column 3 function __memcpy_impl
Symex completed in: 0.019s (257 assignments)
Slicing time: 0.002s (removed 244 assignments)
Generated 18 VCC(s), 7 remaining after simplification (13 assignments)
No solver specified; defaulting to Boolector
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.003s
Solving with solver Boolector 3.2.4
Runtime decision procedure: 0.046s
The forward condition is unable to prove the property
Checking base case, k = 2
Starting Bounded Model Checking
Unwinding loop 25 iteration 1   file /home/bruno/projects/esbmc/src/c2goto/library/string.c line 278 column 3 function __memcpy_impl
Not unwinding loop 25 iteration 2   file /home/bruno/projects/esbmc/src/c2goto/library/string.c line 278 column 3 function __memcpy_impl
Symex completed in: 0.023s (262 assignments)
Caching time: 0.002s (removed 2 assertions)
Slicing time: 0.002s (removed 247 assignments)
Generated 23 VCC(s), 6 remaining after simplification (13 assignments)
No solver specified; defaulting to Boolector
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.003s
Solving with solver Boolector 3.2.4
Runtime decision procedure: 0.000s
BMC program time: 0.027s
No bug has been found in the base case
Checking forward condition, k = 2
Starting Bounded Model Checking
Unwinding loop 25 iteration 1   file /home/bruno/projects/esbmc/src/c2goto/library/string.c line 278 column 3 function __memcpy_impl
Not unwinding loop 25 iteration 2   file /home/bruno/projects/esbmc/src/c2goto/library/string.c line 278 column 3 function __memcpy_impl
Symex completed in: 0.021s (262 assignments)
Slicing time: 0.002s (removed 247 assignments)
Generated 24 VCC(s), 9 remaining after simplification (15 assignments)
No solver specified; defaulting to Boolector
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.003s
Solving with solver Boolector 3.2.4
Runtime decision procedure: 0.043s
The forward condition is unable to prove the property
Checking base case, k = 3
Starting Bounded Model Checking

...

Unwinding loop 25 iteration 8   file /home/bruno/projects/esbmc/src/c2goto/library/string.c line 278 column 3 function __memcpy_impl
Not unwinding loop 43 iteration 9   file main.py line 5 column 12 function kth
terminate called after throwing an instance of 'type2t::symbolic_type_excp'
[1]    6629 IOT instruction (core dumped)  esbmc main.py --incremental-bmc
```

### lcs_length/main.py:
```bash
esbmc main.py
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:5: error: Need type annotation for "dp"  [var-annotated]
main.py:14: error: Call to untyped function "lcs_length" in typed context  [no-untyped-call]
main.py:15: error: Call to untyped function "lcs_length" in typed context  [no-untyped-call]
main.py:16: error: Call to untyped function "lcs_length" in typed context  [no-untyped-call]
main.py:17: error: Call to untyped function "lcs_length" in typed context  [no-untyped-call]
main.py:18: error: Call to untyped function "lcs_length" in typed context  [no-untyped-call]
main.py:19: error: Call to untyped function "lcs_length" in typed context  [no-untyped-call]
main.py:20: error: Call to untyped function "lcs_length" in typed context  [no-untyped-call]
main.py:21: error: Call to untyped function "lcs_length" in typed context  [no-untyped-call]
main.py:22: error: Call to untyped function "lcs_length" in typed context  [no-untyped-call]
Found 11 errors in 1 file (checked 1 source file)

ERROR: Function "Counter" not found (main.py line 5)
```

### levenshtein/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:7: error: Call to untyped function "levenshtein" in typed context  [no-untyped-call]
main.py:11: error: Call to untyped function "levenshtein" in typed context  [no-untyped-call]
main.py:12: error: Call to untyped function "levenshtein" in typed context  [no-untyped-call]
main.py:13: error: Call to untyped function "levenshtein" in typed context  [no-untyped-call]
main.py:17: error: Call to untyped function "levenshtein" in typed context  [no-untyped-call]
main.py:18: error: Call to untyped function "levenshtein" in typed context  [no-untyped-call]
main.py:19: error: Call to untyped function "levenshtein" in typed context  [no-untyped-call]
main.py:24: error: Call to untyped function "levenshtein" in typed context  [no-untyped-call]
main.py:25: error: Call to untyped function "levenshtein" in typed context  [no-untyped-call]
main.py:26: error: Call to untyped function "levenshtein" in typed context  [no-untyped-call]
Found 11 errors in 1 file (checked 1 source file)

Converting
ERROR: min() with more than 2 arguments not yet supported
```

### lis/main.py:
```bash
esbmc main.py
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:3: error: Need type annotation for "ends" (hint: "ends: dict[<type>, <type>] = ...")  [var-annotated]
main.py:36: error: Call to untyped function "lis" in typed context  [no-untyped-call]
main.py:37: error: Call to untyped function "lis" in typed context  [no-untyped-call]
main.py:38: error: Call to untyped function "lis" in typed context  [no-untyped-call]
main.py:39: error: Call to untyped function "lis" in typed context  [no-untyped-call]
main.py:40: error: Call to untyped function "lis" in typed context  [no-untyped-call]
main.py:41: error: Call to untyped function "lis" in typed context  [no-untyped-call]
main.py:42: error: Call to untyped function "lis" in typed context  [no-untyped-call]
main.py:43: error: Call to untyped function "lis" in typed context  [no-untyped-call]
main.py:44: error: Call to untyped function "lis" in typed context  [no-untyped-call]
main.py:45: error: Call to untyped function "lis" in typed context  [no-untyped-call]
main.py:46: error: Call to untyped function "lis" in typed context  [no-untyped-call]
main.py:47: error: Call to untyped function "lis" in typed context  [no-untyped-call]
Found 14 errors in 1 file (checked 1 source file)

WARNING: Empty or malformed list literal detected. Using 'list[int]' as default (main.py:1)
WARNING: Conditional expression has different types in branches: Any vs int. Using 'Any' as fallback (main.py:10)
Converting
ERROR: Type undefined for "ends"
```

### longest_common_subsequence/main.py:
```bash
esbmc main.py
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:7: error: Call to untyped function "longest_common_subsequence" in typed context  [no-untyped-call]
main.py:11: error: Call to untyped function "longest_common_subsequence" in typed context  [no-untyped-call]
main.py:12: error: Call to untyped function "longest_common_subsequence" in typed context  [no-untyped-call]
main.py:16: error: Call to untyped function "longest_common_subsequence" in typed context  [no-untyped-call]
main.py:17: error: Call to untyped function "longest_common_subsequence" in typed context  [no-untyped-call]
main.py:18: error: Call to untyped function "longest_common_subsequence" in typed context  [no-untyped-call]
main.py:19: error: Call to untyped function "longest_common_subsequence" in typed context  [no-untyped-call]
main.py:20: error: Call to untyped function "longest_common_subsequence" in typed context  [no-untyped-call]
main.py:21: error: Call to untyped function "longest_common_subsequence" in typed context  [no-untyped-call]
main.py:22: error: Call to untyped function "longest_common_subsequence" in typed context  [no-untyped-call]
main.py:23: error: Call to untyped function "longest_common_subsequence" in typed context  [no-untyped-call]
main.py:24: error: Call to untyped function "longest_common_subsequence" in typed context  [no-untyped-call]
main.py:25: error: Call to untyped function "longest_common_subsequence" in typed context  [no-untyped-call]
Found 14 errors in 1 file (checked 1 source file)

Converting
Generating GOTO Program
WARNING: function should not return value

* file: main.py
* line: 4
* function: longest_common_subsequence
* column: 8
ERROR:
* type:
migrate expr failed
[1]    14744 IOT instruction (core dumped)  esbmc main.py
```

### mergesort/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:3: error: Function is missing a type annotation  [no-untyped-def]
main.py:21: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:22: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:23: error: Call to untyped function "merge" in typed context  [no-untyped-call]
main.py:96: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:97: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:98: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:99: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:100: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:101: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:102: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:103: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:104: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:105: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:106: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:107: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:108: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
main.py:109: error: Call to untyped function "mergesort" in typed context  [no-untyped-call]
Found 19 errors in 1 file (checked 1 source file)

WARNING: Empty or malformed list literal detected. Using 'list[int]' as default (main.py:0)
Converting
ERROR: All parameters in function "merge" must be type annotated
```

### minimum_spanning_tree/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:3: error: Need type annotation for "group_by_node" (hint: "group_by_node: dict[<type>, <type>] = ...")  [var-annotated]
main.py:16: error: Function is missing a return type annotation  [no-untyped-def]
main.py:16: note: Use "-> None" if function does not return a value
main.py:21: error: Call to untyped function "minimum_spanning_tree" in typed context  [no-untyped-call]
main.py:33: error: Function is missing a return type annotation  [no-untyped-def]
main.py:33: note: Use "-> None" if function does not return a value
main.py:38: error: Call to untyped function "minimum_spanning_tree" in typed context  [no-untyped-call]
main.py:56: error: Function is missing a return type annotation  [no-untyped-def]
main.py:56: note: Use "-> None" if function does not return a value
main.py:61: error: Call to untyped function "minimum_spanning_tree" in typed context  [no-untyped-call]
main.py:71: error: Call to untyped function "test1" in typed context  [no-untyped-call]
main.py:72: error: Call to untyped function "test2" in typed context  [no-untyped-call]
main.py:73: error: Call to untyped function "test3" in typed context  [no-untyped-call]
Found 11 errors in 1 file (checked 1 source file)

ERROR: Type inference failed at line 12. Variable group_by_node not found
```

### next_palindrome/main.py:
```bash
esbmc main.py
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:1: error: Function is missing a type annotation  [no-untyped-def]
main.py:17: error: Call to untyped function "next_palindrome" in typed context  [no-untyped-call]
main.py:18: error: Call to untyped function "next_palindrome" in typed context  [no-untyped-call]
main.py:19: error: Call to untyped function "next_palindrome" in typed context  [no-untyped-call]
main.py:20: error: Call to untyped function "next_palindrome" in typed context  [no-untyped-call]
main.py:21: error: Call to untyped function "next_palindrome" in typed context  [no-untyped-call]
Found 6 errors in 1 file (checked 1 source file)

Converting
ERROR: Subscript assignment not supported in compound assignment
```

### next_permutation/main.py:
```bash
esbmc main.py
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:24: error: Call to untyped function "next_permutation" in typed context  [no-untyped-call]
main.py:25: error: Call to untyped function "next_permutation" in typed context  [no-untyped-call]
main.py:26: error: Call to untyped function "next_permutation" in typed context  [no-untyped-call]
main.py:27: error: Call to untyped function "next_permutation" in typed context  [no-untyped-call]
main.py:28: error: Call to untyped function "next_permutation" in typed context  [no-untyped-call]
main.py:29: error: Call to untyped function "next_permutation" in typed context  [no-untyped-call]
main.py:30: error: Call to untyped function "next_permutation" in typed context  [no-untyped-call]
main.py:31: error: Call to untyped function "next_permutation" in typed context  [no-untyped-call]
Found 9 errors in 1 file (checked 1 source file)

Converting
WARNING: Undefined function 'reversed' - replacing with assert(false)
ERROR: List slicing with missing bounds not yet supported
```

### pascal/main.py:
```bash
esbmc main.py
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:14: error: Call to untyped function "pascal" in typed context  [no-untyped-call]
main.py:15: error: Call to untyped function "pascal" in typed context  [no-untyped-call]
main.py:16: error: Call to untyped function "pascal" in typed context  [no-untyped-call]
main.py:17: error: Call to untyped function "pascal" in typed context  [no-untyped-call]
main.py:18: error: Call to untyped function "pascal" in typed context  [no-untyped-call]
Found 6 errors in 1 file (checked 1 source file)

esbmc: /home/bruno/projects/esbmc/build/_deps/json-src/include/nlohmann/json.hpp:2147: const value_type& nlohmann::json_abi_v3_11_3::basic_json<ObjectType, ArrayType, StringType, BooleanType, NumberIntegerType, NumberUnsignedType, NumberFloatType, AllocatorType, JSONSerializer, BinaryType, CustomBaseClass>::operator[](const typename nlohmann::json_abi_v3_11_3::basic_json<ObjectType, ArrayType, StringType, BooleanType, NumberIntegerType, NumberUnsignedType, NumberFloatType, AllocatorType, JSONSerializer, BinaryType, CustomBaseClass>::object_t::key_type&) const [with ObjectType = std::map; ArrayType = std::vector; StringType = std::__cxx11::basic_string<char>; BooleanType = bool; NumberIntegerType = long int; NumberUnsignedType = long unsigned int; NumberFloatType = double; AllocatorType = std::allocator; JSONSerializer = nlohmann::json_abi_v3_11_3::adl_serializer; BinaryType = std::vector<unsigned char>; CustomBaseClass = void; nlohmann::json_abi_v3_11_3::basic_json<ObjectType, ArrayType, StringType, BooleanType, NumberIntegerType, NumberUnsignedType, NumberFloatType, AllocatorType, JSONSerializer, BinaryType, CustomBaseClass>::const_reference = const nlohmann::json_abi_v3_11_3::basic_json<>&; typename nlohmann::json_abi_v3_11_3::basic_json<ObjectType, ArrayType, StringType, BooleanType, NumberIntegerType, NumberUnsignedType, NumberFloatType, AllocatorType, JSONSerializer, BinaryType, CustomBaseClass>::object_t::key_type = std::__cxx11::basic_string<char>; nlohmann::json_abi_v3_11_3::basic_json<ObjectType, ArrayType, StringType, BooleanType, NumberIntegerType, NumberUnsignedType, NumberFloatType, AllocatorType, JSONSerializer, BinaryType, CustomBaseClass>::object_t = std::map<std::__cxx11::basic_string<char>, nlohmann::json_abi_v3_11_3::basic_json<>, std::less<void>, std::allocator<std::pair<const std::__cxx11::basic_string<char>, nlohmann::json_abi_v3_11_3::basic_json<> > > >]: Assertion `it != m_data.m_value.object->end()' failed.
[1]    27245 IOT instruction (core dumped)  esbmc main.py
```

### possible_change/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:9: error: Call to untyped function "possible_change" in typed context  [no-untyped-call]
main.py:61: error: Call to untyped function "possible_change" in typed context  [no-untyped-call]
main.py:62: error: Call to untyped function "possible_change" in typed context  [no-untyped-call]
main.py:63: error: Call to untyped function "possible_change" in typed context  [no-untyped-call]
main.py:64: error: Call to untyped function "possible_change" in typed context  [no-untyped-call]
main.py:65: error: Call to untyped function "possible_change" in typed context  [no-untyped-call]
main.py:66: error: Call to untyped function "possible_change" in typed context  [no-untyped-call]
main.py:67: error: Call to untyped function "possible_change" in typed context  [no-untyped-call]
main.py:68: error: Call to untyped function "possible_change" in typed context  [no-untyped-call]
main.py:69: error: Call to untyped function "possible_change" in typed context  [no-untyped-call]
main.py:70: error: Call to untyped function "possible_change" in typed context  [no-untyped-call]
Found 12 errors in 1 file (checked 1 source file)

ERROR: Function "possible_change" not found (main.py line 9)
```

### powerset/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:5: error: Call to untyped function "powerset" in typed context  [no-untyped-call]
main.py:20: error: Call to untyped function "powerset" in typed context  [no-untyped-call]
main.py:31: error: Call to untyped function "powerset" in typed context  [no-untyped-call]
main.py:38: error: Call to untyped function "powerset" in typed context  [no-untyped-call]
main.py:43: error: Call to untyped function "powerset" in typed context  [no-untyped-call]
main.py:47: error: Call to untyped function "powerset" in typed context  [no-untyped-call]
Found 7 errors in 1 file (checked 1 source file)

WARNING: Empty or malformed list literal detected. Using 'list[int]' as default (main.py:0)
WARNING: Type inference conflict for parameter 0: list[str] vs list[int]. Using 'int' as fallback (main.py:0)
Converting
ERROR: Cannot unpack signedbv - only tuples and arrays can be unpacked
```

### quicksort/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:7: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:8: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:22: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:23: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:24: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:25: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:26: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:29: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:32: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:35: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:38: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:39: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:40: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:42: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
main.py:45: error: Call to untyped function "quicksort" in typed context  [no-untyped-call]
Found 16 errors in 1 file (checked 1 source file)

ERROR: Type inference failed for Assign at line 7
```

### reverse_linked_list/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
node.py:2: error: Function is missing a type annotation  [no-untyped-def]
node.py:4: error: Cannot assign to a method  [method-assign]
node.py:5: error: Cannot assign to a method  [method-assign]
node.py:6: error: Cannot assign to a method  [method-assign]
node.py:10: error: Function is missing a return type annotation  [no-untyped-def]
node.py:13: error: Function is missing a return type annotation  [no-untyped-def]
node.py:16: error: Function is missing a return type annotation  [no-untyped-def]
main.py:4: error: Function is missing a type annotation  [no-untyped-def]
main.py:32: error: Function is missing a return type annotation  [no-untyped-def]
main.py:32: note: Use "-> None" if function does not return a value
main.py:37: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:38: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:39: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:40: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:41: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:43: error: Call to untyped function "reverse_linked_list" in typed context  [no-untyped-call]
main.py:53: error: Function is missing a return type annotation  [no-untyped-def]
main.py:53: note: Use "-> None" if function does not return a value
main.py:58: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:60: error: Call to untyped function "reverse_linked_list" in typed context  [no-untyped-call]
main.py:70: error: Function is missing a return type annotation  [no-untyped-def]
main.py:70: note: Use "-> None" if function does not return a value
main.py:75: error: Call to untyped function "reverse_linked_list" in typed context  [no-untyped-call]
main.py:84: error: Call to untyped function "test1" in typed context  [no-untyped-call]
main.py:85: error: Call to untyped function "test2" in typed context  [no-untyped-call]
main.py:86: error: Call to untyped function "test3" in typed context  [no-untyped-call]
Found 23 errors in 2 files (checked 1 source file)

ERROR: Type inference failed at line 10. Variable nextnode not found
```

### rpn_eval/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:3: error: Function is missing a type annotation  [no-untyped-def]
main.py:20: error: Call to untyped function "op" in typed context  [no-untyped-call]
main.py:50: error: Call to untyped function "rpn_eval" in typed context  [no-untyped-call]
main.py:51: error: Call to untyped function "rpn_eval" in typed context  [no-untyped-call]
main.py:52: error: Call to untyped function "rpn_eval" in typed context  [no-untyped-call]
main.py:53: error: Call to untyped function "rpn_eval" in typed context  [no-untyped-call]
main.py:54: error: Call to untyped function "rpn_eval" in typed context  [no-untyped-call]
main.py:55: error: Call to untyped function "rpn_eval" in typed context  [no-untyped-call]
Found 9 errors in 1 file (checked 1 source file)

WARNING: Mixed types detected in list literal: float vs str. Using 'list[int]' as fallback (main.py:0)
WARNING: Mixed types detected in list literal: float vs str. Using 'list[int]' as fallback (main.py:0)
WARNING: Mixed types detected in list literal: float vs str. Using 'list[int]' as fallback (main.py:0)
WARNING: Mixed types detected in list literal: float vs str. Using 'list[int]' as fallback (main.py:0)
WARNING: Mixed types detected in list literal: float vs str. Using 'list[int]' as fallback (main.py:0)
WARNING: Mixed types detected in list literal: float vs str. Using 'list[int]' as fallback (main.py:0)
Converting
ERROR: All parameters in function "op" must be type annotated
```

### shortest_path_length/main.py:
```bash
```

### shortest_path_lengths/main.py:
```bash
```

### shortest_paths/main.py:
```bash
```

### shunting_yard/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:3: error: Function is missing a type annotation  [no-untyped-def]
main.py:20: error: Call to untyped function "op" in typed context  [no-untyped-call]
main.py:50: error: Call to untyped function "rpn_eval" in typed context  [no-untyped-call]
main.py:51: error: Call to untyped function "rpn_eval" in typed context  [no-untyped-call]
main.py:52: error: Call to untyped function "rpn_eval" in typed context  [no-untyped-call]
main.py:53: error: Call to untyped function "rpn_eval" in typed context  [no-untyped-call]
main.py:54: error: Call to untyped function "rpn_eval" in typed context  [no-untyped-call]
main.py:55: error: Call to untyped function "rpn_eval" in typed context  [no-untyped-call]
Found 9 errors in 1 file (checked 1 source file)

WARNING: Mixed types detected in list literal: float vs str. Using 'list[int]' as fallback (main.py:0)
WARNING: Mixed types detected in list literal: float vs str. Using 'list[int]' as fallback (main.py:0)
WARNING: Mixed types detected in list literal: float vs str. Using 'list[int]' as fallback (main.py:0)
WARNING: Mixed types detected in list literal: float vs str. Using 'list[int]' as fallback (main.py:0)
WARNING: Mixed types detected in list literal: float vs str. Using 'list[int]' as fallback (main.py:0)
WARNING: Mixed types detected in list literal: float vs str. Using 'list[int]' as fallback (main.py:0)
Converting
ERROR: All parameters in function "op" must be type annotated
```

### sieve/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:3: error: Need type annotation for "primes" (hint: "primes: list[<type>] = ...")  [var-annotated]
main.py:33: error: Call to untyped function "sieve" in typed context  [no-untyped-call]
main.py:34: error: Call to untyped function "sieve" in typed context  [no-untyped-call]
main.py:35: error: Call to untyped function "sieve" in typed context  [no-untyped-call]
main.py:36: error: Call to untyped function "sieve" in typed context  [no-untyped-call]
main.py:37: error: Call to untyped function "sieve" in typed context  [no-untyped-call]
main.py:38: error: Call to untyped function "sieve" in typed context  [no-untyped-call]
Found 8 errors in 1 file (checked 1 source file)

Converting
WARNING: Undefined function 'all' - replacing with assert(false)
esbmc: /home/bruno/projects/esbmc/src/python-frontend/python_list.cpp:961: exprt python_list::compare(const exprt&, const exprt&, const string&): Assertion `lhs_symbol' failed.
[1]    3925 IOT instruction (core dumped)  esbmc main.py
```

### sqrt/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:16: error: Call to untyped function "sqrt" in typed context  [no-untyped-call]
main.py:17: error: Call to untyped function "sqrt" in typed context  [no-untyped-call]
main.py:18: error: Call to untyped function "sqrt" in typed context  [no-untyped-call]
main.py:19: error: Call to untyped function "sqrt" in typed context  [no-untyped-call]
main.py:22: error: Call to untyped function "sqrt" in typed context  [no-untyped-call]
Found 6 errors in 1 file (checked 1 source file)

Converting
Generating GOTO Program
GOTO program creation time: 1.147s
GOTO program processing time: 0.014s
Starting Bounded Model Checking
Symex completed in: 0.000s (6 assignments)
Caching time: 0.002s (removed 1 assertions)
Slicing time: 0.000s (removed 1 assignments)
Generated 5 VCC(s), 4 remaining after simplification (4 assignments)
No solver specified; defaulting to Boolector
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
esbmc: /home/bruno/projects/esbmc/src/solvers/smt/smt_conv.cpp:877: const smt_ast* smt_convt::convert_ast(const expr2tc&): Assertion `is_floatbv_type(expr)' failed.
[1]    5781 IOT instruction (core dumped)  esbmc main.py
```

### subsequences/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:6: error: Need type annotation for "ret" (hint: "ret: list[<type>] = ...")  [var-annotated]
main.py:9: error: Call to untyped function "subsequences" in typed context  [no-untyped-call]
main.py:14: error: Call to untyped function "subsequences" in typed context  [no-untyped-call]
main.py:15: error: Call to untyped function "subsequences" in typed context  [no-untyped-call]
main.py:16: error: Call to untyped function "subsequences" in typed context  [no-untyped-call]
main.py:17: error: Call to untyped function "subsequences" in typed context  [no-untyped-call]
main.py:22: error: Call to untyped function "subsequences" in typed context  [no-untyped-call]
main.py:23: error: Call to untyped function "subsequences" in typed context  [no-untyped-call]
main.py:33: error: Call to untyped function "subsequences" in typed context  [no-untyped-call]
main.py:63: error: Call to untyped function "subsequences" in typed context  [no-untyped-call]
main.py:106: error: Call to untyped function "subsequences" in typed context  [no-untyped-call]
main.py:107: error: Call to untyped function "subsequences" in typed context  [no-untyped-call]
main.py:112: error: Call to untyped function "subsequences" in typed context  [no-untyped-call]
main.py:113: error: Call to untyped function "subsequences" in typed context  [no-untyped-call]
Found 15 errors in 1 file (checked 1 source file)

Converting
ERROR: Unsupported expression GeneratorExp at line 8
```

### to_base/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:3: error: Function is missing a type annotation  [no-untyped-def]
main.py:24: error: Call to untyped function "to_base" in typed context  [no-untyped-call]
main.py:25: error: Call to untyped function "to_base" in typed context  [no-untyped-call]
main.py:26: error: Call to untyped function "to_base" in typed context  [no-untyped-call]
main.py:27: error: Call to untyped function "to_base" in typed context  [no-untyped-call]
main.py:28: error: Call to untyped function "to_base" in typed context  [no-untyped-call]
main.py:29: error: Call to untyped function "to_base" in typed context  [no-untyped-call]
main.py:30: error: Call to untyped function "to_base" in typed context  [no-untyped-call]
main.py:31: error: Call to untyped function "to_base" in typed context  [no-untyped-call]
main.py:32: error: Call to untyped function "to_base" in typed context  [no-untyped-call]
main.py:33: error: Call to untyped function "to_base" in typed context  [no-untyped-call]
Found 11 errors in 1 file (checked 1 source file)

ERROR: Type inference failed for Assign at line 5
```

### topological_ordering/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
node.py:2: error: Function is missing a type annotation  [no-untyped-def]
node.py:4: error: Cannot assign to a method  [method-assign]
node.py:5: error: Cannot assign to a method  [method-assign]
node.py:6: error: Cannot assign to a method  [method-assign]
node.py:10: error: Function is missing a return type annotation  [no-untyped-def]
node.py:13: error: Function is missing a return type annotation  [no-untyped-def]
node.py:16: error: Function is missing a return type annotation  [no-untyped-def]
main.py:3: error: Function is missing a type annotation  [no-untyped-def]
main.py:13: error: Function is missing a return type annotation  [no-untyped-def]
main.py:13: note: Use "-> None" if function does not return a value
main.py:18: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:19: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:20: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:21: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:22: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:23: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:24: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:25: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:40: error: Call to untyped function "topological_ordering" in typed context  [no-untyped-call]
main.py:48: error: Function is missing a return type annotation  [no-untyped-def]
main.py:48: note: Use "-> None" if function does not return a value
main.py:53: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:54: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:55: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:56: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:57: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:58: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:70: error: Call to untyped function "topological_ordering" in typed context  [no-untyped-call]
main.py:76: error: Function is missing a return type annotation  [no-untyped-def]
main.py:76: note: Use "-> None" if function does not return a value
main.py:79: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:80: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:81: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:82: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:83: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:84: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:85: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:86: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:87: error: Call to untyped function "Node" in typed context  [no-untyped-call]
main.py:105: error: Call to untyped function "topological_ordering" in typed context  [no-untyped-call]
main.py:123: error: Call to untyped function "test1" in typed context  [no-untyped-call]
main.py:124: error: Call to untyped function "test2" in typed context  [no-untyped-call]
main.py:125: error: Call to untyped function "test3" in typed context  [no-untyped-call]
Found 40 errors in 2 files (checked 1 source file)

WARNING: Could not determine list element type. Using 'list[int]' as fallback (main.py:0)
WARNING: Could not determine list element type. Using 'list[int]' as fallback (main.py:0)
WARNING: Could not determine list element type. Using 'list[int]' as fallback (main.py:0)
Converting
ERROR: All parameters in function "Node" must be type annotated
```

### wrap/main.py:
```bash
ESBMC version 7.11.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py

Type checking warning:
main.py:2: error: Function is missing a type annotation  [no-untyped-def]
main.py:31: error: Call to untyped function "wrap" in typed context  [no-untyped-call]
main.py:54: error: Call to untyped function "wrap" in typed context  [no-untyped-call]
main.py:113: error: Call to untyped function "wrap" in typed context  [no-untyped-call]
main.py:129: error: Call to untyped function "wrap" in typed context  [no-untyped-call]
main.py:145: error: Call to untyped function "wrap" in typed context  [no-untyped-call]
Found 6 errors in 1 file (checked 1 source file)

[1]    12992 segmentation fault (core dumped)  esbmc main.py
```