---
title: Usage
weight: 2
---

## Prerequisites

ESBMC must be built with the Solidity frontend enabled:

```bash
cmake -GNinja -Bbuild -S . \
  -DDOWNLOAD_DEPENDENCIES=On \
  -DENABLE_SOLIDITY_FRONTEND=On \
  -DENABLE_Z3=On \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja -C build
```

The Solidity compiler `solc` (≥ 0.5.0; ≥ 0.7 recommended) must be reachable through one of:

- `--solc-bin /path/to/solc`
- the `$SOLC` environment variable
- `solc` on `$PATH`

## Basic Invocation

Pass a `.sol` file directly:

```bash
esbmc contract.sol
```

If the source declares more than one contract, name the entry point with `--contract`:

```bash
esbmc contract.sol --contract MyContract
```

You can also pre-compile the AST and pass it explicitly:

```bash
solc --ast-compact-json contract.sol > contract.solast
esbmc --sol contract.sol contract.solast --contract MyContract
```

## Useful Flags

Solidity-specific flags (a full list is available via `esbmc --help`):

| Flag | Description |
|---|---|
| `--sol <path>` | Solidity source file (also accepted as a positional argument) |
| `--contract <name>` | Set contract name; required when the source declares more than one contract |
| `--solc-bin <path>` | Path to the solc binary (default: `$SOLC` or `solc` on `$PATH`) |
| `--focus-function <name>` | Restrict the dispatcher loop to one function. The constructor still runs. |
| `--no-visibility` | Verify every function, including unreachable internal/private ones |
| `--unbound` | Model external function calls as arbitrary behaviour (default) |
| `--bound` | Model inter-contract function calls within a bounded system |
| `--reentry-check` | Detect reentrancy behaviour during contract execution |

Common ESBMC flags that work with Solidity:

| Flag | Description |
|---|---|
| `--unwind N` | Set the global loop unwind bound to `N` |
| `--incremental-bmc` | Increase the unwind bound incrementally until a bug is found or the maximum is reached |
| `--k-induction` | Use the k-induction proof rule (full correctness proofs) |
| `--max-k-step N` | Bound for k-induction iteration |
| `--multi-property` | Continue verification after the first failure, reporting all violated properties |
| `--overflow-check` | Enable arithmetic overflow / underflow checks |
| `--bitwuzla` / `--z3` / `--cvc5` | Pick a specific SMT solver |

## Writing Verification Harnesses

You write properties directly inside contract functions using `assert`, `__ESBMC_assume`, and `__VERIFIER_nondet_*`:

```solidity
pragma solidity >=0.8.0;

contract Counter {
    uint public x;

    function add(uint n) public {
        x = x + n;
    }

    function inv() public view {
        assert(x >= 0);   // checked after every dispatcher iteration
    }
}
```

You don't write a `main` function — the frontend auto-generates a dispatcher loop that:

1. Runs the constructor.
2. Iterates a non-deterministic loop, picking a random public function and random arguments each time.
3. Re-checks every `assert` after each call.

Use `--focus-function inv` to restrict the dispatcher to a single function while still running the constructor.

---

## Examples

### Example 1: Out-of-Bounds Array Access

```solidity
pragma solidity >=0.4.26;

contract MyContract {
    function func_array_loop() external pure {
        uint8[2] memory a;
        a[0] = 100;
        for (uint8 i = 1; i < 3; ++i) {
            a[i] = 100;
            assert(a[i-1] == 100);
        }
    }
}
```

```bash
esbmc contract.sol --contract MyContract --incremental-bmc
```

```
Violated property:
  array bounds violated: array `a' upper bound
  (signed long int)i < 2

VERIFICATION FAILED
```

The third loop iteration writes `a[2]` on a length-2 array. ESBMC catches the out-of-bounds write before the assertion check fires.

---

### Example 2: State Invariant Across Multiple Calls

A property that depends on the *sequence* of calls — typical for stateful smart contracts:

```solidity
pragma solidity >=0.8.0;

contract Robot {
    int x = 0;
    int y = 0;

    function moveLeftUp()    public { --x; ++y; }
    function moveLeftDown()  public { --x; --y; }
    function moveRightUp()   public { ++x; ++y; }
    function moveRightDown() public { ++x; --y; }

    function inv() public view {
        assert((x + y) % 2 != 0);   // false after any single move
    }
}
```

```bash
esbmc contract.sol --contract Robot --k-induction
```

```
[Counterexample]
State 1: x = -1
State 2: y = 1
...
Violated property: assertion (x + y) % 2 != 0

VERIFICATION FAILED
```

ESBMC discovers a sequence of calls that drives `(x + y) % 2 == 0`, refuting the invariant.

---

### Example 3: Integer Overflow

```solidity
pragma solidity >=0.8.0;

contract Adder {
    uint256 total;

    function add(uint256 a, uint256 b) public {
        unchecked {
            total = a + b;        // wraps on overflow
        }
        assert(total >= a);       // can be false
    }
}
```

```bash
esbmc contract.sol --contract Adder --overflow-check
```

```
Violated property:
  arithmetic overflow on add

VERIFICATION FAILED
```

The `unchecked` block disables Solidity's built-in overflow trap; `--overflow-check` re-introduces the check at the verifier level.

---

### Example 4: Reentrancy Detection

```solidity
pragma solidity >=0.8.0;

interface IReceiver { function onReceived() external; }

contract Vault {
    mapping(address => uint) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() external {
        uint amount = balances[msg.sender];
        IReceiver(msg.sender).onReceived();   // external call before state update
        balances[msg.sender] = 0;
        payable(msg.sender).transfer(amount);
    }
}
```

```bash
esbmc contract.sol --contract Vault --reentry-check --bound
```

ESBMC explores reentrant call sequences and reports a counterexample where a malicious receiver calls `withdraw()` recursively before the balance is zeroed.

---

### Example 5: Multi-Contract Bounded Verification

```solidity
pragma solidity >=0.8.0;

contract Vault {
    constructor() payable {}
    function withdraw(address payable to, uint256 amt) external {
        to.transfer(amt);
    }
}

contract Probe {
    function check() external payable {
        uint256 pre  = address(this).balance;
        Vault v = new Vault{value: 100}();
        v.withdraw(payable(address(this)), 100);
        uint256 post = address(this).balance;
        assert(pre == post);
    }
}
```

```bash
esbmc contract.sol --contract Probe --bound --unwind 4
```

`--bound` wires inter-contract calls into the same verification run instead of treating them as arbitrary external behaviour, so balance flows can be tracked across contract boundaries.
