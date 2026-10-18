# msg.sender pinning on a contract that also makes value transfers

## What this test pins

CityToken guards `setCEO` / `setCOO` / `createToken` / `payout` /
`withdrawFunds` with modifiers that do `require(msg.sender == ceoAddress)`,
`require(msg.sender == cooAddress)` or the `onlyCLevel` disjunction. ESBMC
covers the authorised (guard-true) side of each, so the generated Foundry test
must pin `msg.sender` with `vm.prank(...)` — otherwise the replay calls arrive
from the test contract, every guarded call reverts, and the authorised branch
is reported covered while no test actually reaches it.

The three `^  !\(!\(...\)\)` regexes are the authorised arms in the
`--branch-coverage-claims` reached list; the `vm.prank` line proves the
generator reproduced the sender that reaches them. Counts are matched loosely
(`[0-9]+`): the exact number of cases follows the solver's model choices, and
the "N call(s) with pinned msg.sender" line is only printed when N > 0, so a
zero-pin regression cannot match it.

## The bug this covers (fixed)

`foundry_generator::reconstruct` refused to pin the sender whenever any SSA
step assigned the `msg_sender` global outside the per-tx reseed, on the theory
that a nested/high/low-level call wrapper had overwritten it and a top-level
`vm.prank` could no longer reproduce what the covered branch read.

That test was syntactic, and symex does not preserve the syntax: an assignment
that occurs under a branch is merged into an *unconditional* SSA step whose RHS
is `cond ? new_value : old_value`. Every `$transfer` / `$send` / `$call`
wrapper in the contract therefore shows up as a guard-true `msg_sender` write
on *every* path — including paths that never enter it. One `.transfer()`
anywhere in the contract was enough to suppress the pin for all 29 cases,
including `setCEO`, which makes no call at all.

The fix decides on the *model value* instead: a write that leaves `msg_sender`
equal to the transaction's top-level sender (an untaken-branch merge, or the
wrapper's own restore on the way out) does not shadow it. Only a read taken
while the value genuinely differs marks the segment un-pinnable — which is what
the original comment intended.

This is why the defect did not reproduce on the minimal variants tried
(inline `require(msg.sender == owner)`, a single-comparison modifier, a mutable
owner, a ctor-bound owner, an inherited guard variable): none of them contained
a value transfer, so no wrapper existed to poison the path.

## Replay shape

The generator emits two test contracts, one per deployer identity the solver
picked (`CityTokenCovTest_0` deploys under `vm.startPrank(address(0))`,
`CityTokenCovTest_1` under `vm.startPrank(address(uint160(4294967295)))`).
Within each group the calls pranked with that group's deployer exercise the
authorised arm and the others the reverting arm, so both sides of every
access-control branch have a reaching test.
