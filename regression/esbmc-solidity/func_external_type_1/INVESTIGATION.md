# func_external_type_1 KNOWNBUG Investigation

## Summary

Frontend crash when converting external function types passed as values
(e.g., `this.oracleResponse` passed to `query()`).
This is a **Solidity frontend bug**, not a backend/symex issue.

## Error

```
terminate called after throwing an instance of 'nlohmann::json_abi_v3_11_3::detail::type_error'
  what():  [json.exception.type_error.305] cannot use operator[] with a string argument with array
```

The crash occurs when the frontend tries to convert `this.oracleResponse` — a MemberAccess
that produces an external function reference used as a value (not as a direct call).

## C Equivalent PoC

The equivalent C program using function pointers in structs works correctly:

```c
typedef void (*callback_t)(unsigned);
struct Request { char data[32]; callback_t callback; };
void query(const char *data, callback_t callback) { ... }
void reply(unsigned requestID, unsigned response) {
    requests[requestID].callback(response);
}
// ... callback correctly invoked => VERIFICATION SUCCESSFUL
```

**Result**: `VERIFICATION SUCCESSFUL` — ESBMC's C backend correctly handles function
pointers stored in structs, passed as arguments, and invoked later.

## Root Cause

The bug is entirely in the **Solidity frontend**. Two features are missing:

1. **Function reference as value**: `this.oracleResponse` produces a function reference
   that must be serialized as a value (address + selector pair in Solidity ABI).
   The frontend crashes when trying to convert this expression.

2. **External function types in struct members**: `function(uint) external callback`
   as a struct field. The frontend doesn't know how to represent this type.

This is a superset of the `func_internal_type_1` problem — it additionally requires
cross-contract function reference serialization.

## Complexity

Very high. Requires function pointer support (same as func_internal_type_1) PLUS
cross-contract function reference modeling (address + selector encoding).

## Date

2026-04-12
