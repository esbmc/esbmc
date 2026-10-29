# Re-entrancy is never allowed to complete, so the violation is missed

FIXED -- this test is CORE again and reports `VERIFICATION FAILED` at k=2 in
about 5s. See "The fix" at the end.

History: `97e37211bc` flipped both `reentrance_2` and `reentrance_13` to CORE
via `--incremental-bmc`; they stopped reproducing their violation and reported
`VERIFICATION UNKNOWN`, so both were re-pinned KNOWNBUG. The regression
predates 2026-07-11 (a static binary from that date behaves identically) and
had been masked by the local 120s regression timeout, which fired before the
UNKNOWN was ever printed. `reentrance_13` is a DIFFERENT bug and stays
KNOWNBUG -- see below.

## Root cause (measured, not inferred)

A re-entrant `Bank.withdraw` frame always gets `success == false` back from its
own `msg.sender.call{value: amount}("")`. It therefore reverts; the revert
propagates out through `$call#1`'s `aux49` (`RETURN !(bool)aux49`) and makes the
OUTER frame's `require(success)` fail too, so the outer frame reverts before
reaching its post-condition. The post-condition is consequently never evaluated
in a state where a nested withdraw has drained the contract -- which is exactly
the state the assertion is meant to catch.

Evidence, using a `uint private depth` counter incremented on entry to
`Bank.withdraw` (`--incremental-bmc --bound --cvc5`, ladder run to k=50):

| probe | assertion | placement | result |
|---|---|---|---|
| 15 | `assert(depth < 1)` | after `require(success)` | FAILED at k=1 — the counter works |
| 16 | `assert(depth < 2)` | after the call, BEFORE `require(success)` | FAILED at k=2 — a nested frame does run its own call and return |
| 14 | `assert(depth < 2)` | after `require(success)` | never violated to k=50 — no nested frame ever passes it |

Two further probes bound the diagnosis:

- Replacing the attacker's unbounded guard with a single-shot one
  (`bool done; if (!done) { done = true; target.withdraw(msg.value); }`) makes
  ESBMC report the REAL violation -- `new_contract_balance ==
  old_contract_balance - amount` -- at k=2. Keeping the balance guard and
  adding the flag (`if (address(target).balance > 0 && !done)`) also finds it.
  So re-entrancy detection itself works; what fails is the unbounded-recursion
  shape.
- Plain BMC with an assumption-based recursion cut
  (`--unwind 3/5 --no-unwinding-assertions`) also finds nothing, so the path is
  genuinely infeasible in the model rather than merely cut away by the bound.

Ingredients that ARE individually reachable (each shown by an `assert(false)` or
inverted-assertion probe): `receive()` is entered; the guarded
`target.withdraw(msg.value)` branch is entered; that call returns normally; the
path after `require(success)` is reachable; `address(this).balance >= 2*amount`
and `== 2*amount` hold at withdraw entry; and Bank's assertions are in the claim
set. Only the COMBINATION "a nested frame completed" + "outer call succeeded" is
unreachable.

Note when reading probes that place `assert(false)` inside `receive()`: it is an
external payable function, so the dispatcher can call it as a TOP-LEVEL
transaction. Such a probe proves top-level reachability, not re-entrancy; the
depth counter above is what distinguishes the nested frame.

## Why the nested call always fails: the handle never sees the drain

The attacker re-enters only while `address(target).balance > 0`, so the
recursion can terminate cleanly only once that balance reaches 0. It never
does. Instrumenting `receive()` as

```solidity
uint before = address(target).balance;
if (before > 0) {
    target.withdraw(msg.value);
    assert(address(target).balance == before);   // never violated, k<=50
}
```

shows the balance read through the interface handle is UNCHANGED after a nested
`withdraw` returns -- even though `$call#1` debits `this->$balance` inside the
callee and `address(this).balance` does change there.

The read and the write land on two DIFFERENT model objects, and the reason is
specific to the handle's declared type being an interface:

- `IBank(_target)` lowers to `&_ESBMC_Object_IBank` -- the singleton of the
  DECLARED type -- and sets `_ESBMC_Object_IBank.$address = _target`
  (`convert_type_expr`, the address->contract branch,
  src/solidity-frontend/solidity_convert_type.cpp:1776-1799). The handle is a
  valid pointer; an earlier draft of this note read a counterexample line as a
  nil pointer, which was wrong.
- `target.withdraw(...)` dispatches on `base->_ESBMC_bind_cname` and selects an
  IMPLEMENTATION singleton, `withdraw(&_ESBMC_Object_Bank, ...)`
  (`get_high_level_member_access`, solidity_convert_call.cpp:1445,1471).
- Interfaces and abstract contracts are skipped everywhere the cname ladder is
  built (solidity_convert_builtin.cpp:495, solidity_convert_call.cpp:2403,
  2840, 3349), so `_ESBMC_Object_IBank` is never a dispatch target.

`address(target).balance` therefore read `_ESBMC_Object_IBank.$balance` -- a
field that NO call in the model can ever write.

So the guard is effectively constant-true, the recursion runs to the unwind
bound on every path, no frame terminates cleanly, and the outer
`require(success)` never passes. That is the single mechanism behind every
observation above, and it also explains why the `bool done` attacker works: its
termination does not depend on observing the drain.

## The fix

`get_builtin_property_expr` (src/solidity-frontend/solidity_convert_builtin.cpp)
used to take `handle->$<prop>` directly for ANY contract-typed base, under a
condition whose own comment read `//TODO: fixme! this pattern match is weak`.
It now does that only when the field really is the account being read:

- `this` -- always the executing instance.
- a CONCRETE contract-typed handle -- both `new C()` and the cast `C(_addr)`
  produce `&_ESBMC_Object_C`, and dispatch selects that same singleton when
  `_ESBMC_bind_cname == C`, so read and write already meet.

An interface- or abstract-typed handle (declared name in `nonContractNamesList`)
now routes through `get_aux_property_function` instead: the existing ladder
resolves the handle's `$address` against every tracked object and falls back to
the EOA balance map -- the account identity EVM itself uses for
`address(x).balance`. A handle whose declared contract name cannot be
determined keeps the historical direct read, so the blast radius is exactly the
interface/abstract case that was reading a dead field.

This restores a path on which `_target == _ESBMC_Object_Bank.$address` and the
guard observes the real drain, so the recursion can terminate, the nested frame
returns `success == 1`, and the outer post-condition is finally evaluated on a
drained state. `VERIFICATION FAILED` at k=2.

## reentrance_13 (also fixed, different cause)

`reentrance_13` is the checks-effects-interactions-correct variant. It was
pinned alongside this test on the assumption of a shared cause; that assumption
was wrong. Its `receive()` had no balance guard at all, so the constant-guard
mechanism above was never what blocked it -- its attacker simply could not
terminate, and the resulting revert cascade meant `success` was always false.
See ../reentrance_13/NOTES.md. It is CORE again with the same guard this test
uses.
