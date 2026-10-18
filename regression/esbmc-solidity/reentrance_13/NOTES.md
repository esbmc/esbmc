# The attacker could not terminate, so it could never witness the bug

FIXED -- CORE again, `VERIFICATION FAILED` at k=2 in about 7s.

This variant is the checks-effects-interactions-CORRECT bank: `Bank.withdraw`
decrements `balances[msg.sender]` BEFORE the external call. Its assertion is a
ghost-variable post-condition, `new_user_balance == old_user_balance - amount`,
which re-entrancy still breaks even though the CEI ordering prevents the
classic drain: the nested frame decrements a second time, so when the outer
frame resumes it observes `old - 2*amount` where it expects `old - amount`.
That is a REAL violation and the point of the benchmark.

## Why it stopped reproducing

The bug is only observable if the outer frame gets past `require(success)`.
`Reproduction.receive()` used to re-enter with NO guard:

```solidity
receive() external payable { target.withdraw(msg.value); }
```

Such an attacker can never stop. The chain
`withdraw -> call -> receive -> withdraw -> ...` decrements by `amount` each
level, so it necessarily ends at some depth n where
`require(amount <= balances[msg.sender])` fails. That revert propagates back
through EVERY frame, because a low-level call returns `ok = !reverted`
(`emit_call_revert_return`, src/solidity-frontend/solidity_convert_call.cpp,
`return !_rev; // ok is false iff the callee reverted`). Each frame's
`require(success)` therefore fails in turn and the whole transaction reverts, so
the post-condition is never evaluated. This is faithful EVM behaviour, not a
modelling defect.

Measured on the unguarded fixture (`--contract Reproduction --no-standard-checks
--incremental-bmc --bound --cvc5`), with a `uint depth` counter incremented on
entry to `withdraw`:

| probe | assertion | placement | result |
|---|---|---|---|
| P1 | `assert(depth < 2)` | after the call, BEFORE `require(success)` | FAILED at k=3 -- a nested frame does run |
| P2 | `assert(depth < 2)` | after `require(success)` | never violated to k=12 -- no nested frame ever gets past it |
| P3 | `assert(!success)` | after the call | never violated to k=10 -- `success` is ALWAYS false |
| P4 | original assert, attacker guarded by a one-shot `bool done` | -- | FAILED at k=2 in 3s -- the violation IS real and reachable |
| P5 | original assert, attacker guarded by `address(target).balance > 0` | -- | FAILED at k=2 in 7s -- adopted as the fix |

P3 is the decisive one: the low-level call in `Bank.withdraw` can never succeed
against an attacker that always re-enters.

## Why it used to pass

`97e37211bc` (2026-05-01) flipped this test to CORE. At that time the low-level
call model hard-coded `ok = true`, so step 4 above never happened: the outer
frame sailed past `require(success)` even though the nested chain had reverted,
and evaluated the post-condition on a double-decremented balance. That FAILED
was an artifact of an unsound call model. `e4d96f8bcc` (2026-06-28,
"model low-level call failure as ok = !reverted under --bound") made the model
faithful and the artifact disappeared, which is what `4219bb1041` re-pinned as
KNOWNBUG.

So the regression was ESBMC becoming MORE correct while the fixture stayed
broken.

## The fix

Give the attacker a termination condition -- the same one `reentrance_2`
already uses:

```solidity
receive() external payable {
    if (address(target).balance > 0) {
        target.withdraw(msg.value);
    }
}
```

Once Bank's ETH balance reaches 0 the attacker returns without re-entering, its
`receive` completes normally, `success` is true all the way back up, and the
outer frame finally evaluates its post-condition on a re-entered state.

`Bank` -- the upstream benchmark contract
(https://github.com/fsainas/contracts-verification-benchmark) -- is
byte-identical. Only `Reproduction`, this repo's own attacker scaffolding,
changed. The two tests are now a matched pair differing solely in where `Bank`
does its bookkeeping: `reentrance_2` decrements after the call, `reentrance_13`
before.

This fix depends on `f2bbfa0406`: the guard reads `address(target).balance`
through an `IBank` handle, and before that commit an interface-typed handle's
`$balance` was a field no call could write, so the guard would have been
constant-true and this attacker would not have terminated either.

## Known limitation this exposes

ESBMC re-enters only through attacker contracts that exist in the source. It
cannot synthesise an arbitrary adversarial fallback, so a contract whose only
in-source caller never stops re-entering will not have its re-entrancy
violations witnessed. `--reentry-check` does report a violation here at k=1, but
of a different property (`!"Reentrancy behavior detected"`), not this ghost
post-condition.
