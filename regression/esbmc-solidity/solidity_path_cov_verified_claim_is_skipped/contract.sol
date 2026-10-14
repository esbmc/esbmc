// A REFUTED path claim must be solved ONCE, not once per physical copy.
//
// Under the default `--unbound` model an external call is a nondet RE-ENTRY
// into this contract's own dispatcher, so `reachOut`'s body is instantiated
// once per re-entry level and every physical copy carries its own copy of the
// unit's exit assertions. One instrumented claim therefore reaches the solve
// loop many times under ONE claim key.
//
// `multi_property_check` has always had the machine to stop that: a set of
// already-refuted claim keys and an early return. It never fired under
// coverage, because the INSERT used `claim_sig` (msg + "\t" + loc) while the
// LOOKUP used `claim_cstr` (msg + " at " + loc) -- two spellings that can never
// be equal. The skip was dead code and every copy was solved again from
// scratch.
//
// MEASURED before the repair, on `EscrowDst --focus-function withdraw`: 5
// distinct claim keys, 425 VCCs, ~85 solves per path. All four obtainable
// witnesses were in hand after 46 solves; the rest re-derived them until the
// process died of `std::bad_alloc` at 8 GiB. With the repair the same run is
// COMPLETE at 100 %.
//
// WHAT THIS TEST PINS, and why each line is here rather than a prettier one:
//
//   * `N copy(ies) of already-witnessed path claim(s) were not solved again`
//     -- printed ONLY when the skip counter is non-zero, and that counter can
//     only be incremented by the branch this repair revives. On the defective
//     build the line is ABSENT entirely, so this pin is RED there for any
//     contract with a duplicated instantiation. It is the whole point of the
//     test.
//
//   * `^Path Status: F 3, I 0, U 0$` -- the soundness half. The skip must not
//     cost coverage. If a future change extends the skip to already-PASSED
//     keys, this line is what catches it: an UNSAT verdict is NOT final across
//     copies (a path that holds at one re-entry depth can be feasible at
//     another -- measured: EscrowDst's fifth path became F on its 85th
//     instantiation after PASSING on the first), so skipping PASSED keys would
//     silently lose paths.
//
// The two pins must both hold. Either alone is satisfiable by a broken build:
// the first alone by a skip that also drops witnesses, the second alone by the
// dead code that was there all along.
pragma solidity ^0.8.0;

contract C {
    uint256 public x;

    function reachOut(address t) public {
        (bool ok, ) = t.call("");
        if (ok) {
            x = 1;
        } else {
            x = 2;
        }
    }
}
