---
title: Supported Features
weight: 3
---

This page is a reference of Solidity language constructs and built-ins currently supported by ESBMC-Solidity.

## Types

### Value Types

- **Booleans**: `bool`
- **Signed integers**: `int8`, `int16`, `int24`, `int32`, … `int256` (every multiple of 8)
- **Unsigned integers**: `uint8`, `uint16`, … `uint256`
- **Address**: `address`, `address payable`
- **Fixed-size byte arrays**: `bytes1`, `bytes2`, … `bytes32`
- **Enums**
- **User-defined value types**: `type T is <elementary>;`
- **Function types**: internal/external function references (modelled as opaque pointers; indirect calls return non-deterministic values)

### Reference Types

- **Dynamic byte arrays**: `bytes`
- **Strings**: `string`
- **Fixed-size arrays**: `T[N]`
- **Dynamic arrays**: `T[]`
- **Multi-dimensional arrays**: `T[][]`, `T[][N]`, `T[N][]`, `T[N][M]`
- **Mappings**: `mapping(K => V)`, including nested mappings
- **Structs**
- **Tuples** (used in destructuring assignments and multi-return)

### Data Locations

- `storage`, `memory`, `calldata` (recognised on parameters, locals, and return values)

## Contracts and Inheritance

- **Contract definitions**, including the contract-as-type pattern (`MyContract c = new MyContract(...)`)
- **Abstract contracts**, **interfaces**, and **libraries**
- **Single and multiple inheritance** with C3 linearisation
- **`super` calls** resolved against the linearised base list
- **`this`** as the implicit receiver
- **Visibility modifiers**: `public`, `private`, `internal`, `external`
- **Function modifiers** (inlined into call sites)
- **Constructors**, **`receive`** functions, **`fallback`** functions, **`payable`** attribute
- **Virtual / override** with linearised dispatch
- **Events** (parsed; treated as no-ops for verification)
- **Custom errors** (parsed; `revert MyError(...)` treated as a verification cut-off)
- **Function overloading**

## Control Flow

- `if` / `else`
- `for`, `while`, `do`–`while`
- `break`, `continue`
- `return` (including multi-value returns into tuples)
- `try` / `catch` (the success path is modelled precisely; revert paths cut off symbolic execution)
- `require(cond)` — modelled as `__ESBMC_assume(cond)`
- `assert(cond)` — modelled as a verification claim
- `revert(msg)` / `revert MyError(...)` — modelled as `__ESBMC_assume(false)`
- `unchecked { ... }` blocks (overflow checks bypassed inside; `--overflow-check` re-enables them)

## Operators

| Category | Operators |
|---|---|
| Arithmetic | `+`, `-`, `*`, `/`, `%`, `**` |
| Bitwise | `&`, `\|`, `^`, `~` |
| Shift | `<<`, `>>` |
| Comparison | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| Logical | `&&`, `\|\|`, `!` |
| Assignment | `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `<<=`, `>>=`, `&=`, `\|=`, `^=` |
| Increment / decrement | `++`, `--` (prefix and postfix) |
| Ternary | `cond ? a : b` |
| Unary | `delete x` |

## Built-in Variables and Functions

### Message and Transaction Context

- `msg.sender`, `msg.value`, `msg.data`
- `tx.origin`, `tx.gasprice`
- `block.number`, `block.timestamp`, `block.difficulty`, `block.coinbase`, `block.gaslimit`, `block.chainid`
- `gasleft()`

These are non-deterministic per call but stable within a single call's body.

### Address Members

- `address(x).balance`
- `payable(addr).transfer(amt)`, `payable(addr).send(amt)`
- `addr.call(data)`, `addr.delegatecall(data)`, `addr.staticcall(data)`
- `address(this)`, `address(c)` for a contract instance `c`

### Cryptographic and Math

- `keccak256(bytes)`
- `sha256(bytes)`
- `ecrecover(hash, v, r, s)`
- `addmod(a, b, m)`, `mulmod(a, b, m)`

### ABI Encoding

- `abi.encode(...)`
- `abi.encodePacked(...)`
- `abi.encodeWithSelector(selector, ...)`
- `abi.encodeWithSignature(sig, ...)`
- `abi.encodeCall(fn, (...))`
- `abi.decode(data, (T1, T2, …))`

### Bytes and String

- `bytes.concat(...)`
- `string.concat(...)`
- `bytes(s).length`, indexing into a `bytes`
- Cast between `string` and `bytes`

### Type Information

- `type(T).min`, `type(T).max`
- `type(T).creationCode`, `type(T).runtimeCode` (modelled as opaque `bytes`)
- `type(T).interfaceId`

### Contract-Related

- `selfdestruct(payable)` — drains the contract's balance to the recipient address
- Contract deployment via `new T(...)` and `new T{value: ...}(...)`

### Ether and Time Units

- Ether: `wei`, `gwei`, `szabo`, `finney`, `ether`
- Time: `seconds`, `minutes`, `hours`, `days`, `weeks`, `years`

## Verification Harness Primitives

You can write properties inline in any contract function:

| Primitive | Meaning |
|---|---|
| `assert(cond)` | Property to verify; failure produces a counterexample |
| `__ESBMC_assume(cond)` | Constrain the symbolic state; paths violating `cond` are pruned |
| `__VERIFIER_assume(cond)` | Alias for `__ESBMC_assume` |
| `__VERIFIER_assert(cond)` | Alias for `assert` |
| `__VERIFIER_nondet_uint()` / `_int()` / `_bool()` / `_address()` / … | Inject a non-deterministic value of the given type |

The frontend automatically wraps every public/external function with a non-deterministic dispatcher loop, so contract state evolves across an arbitrary number of calls. Use `--focus-function <name>` to restrict the loop to a specific function.
