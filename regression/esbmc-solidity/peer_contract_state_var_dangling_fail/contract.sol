// Worked example #2 documenting --bound semantics for peer contract instances.
//
// Pair with peer_contract_explicit_new_pass: same Inner contract, same assertion,
// only difference is that `inner` here is a state variable declared at contract
// scope WITHOUT explicit construction. Under --bound the harness creates the
// _ESBMC_Object_Inner peer instance (and runs Inner's constructor on it), but
// `Outer.inner` itself stays as an uninitialised state-var reference — exactly
// like real Solidity, where an unassigned contract-typed state var points to
// address(0). Calling `inner.sig()` therefore does NOT read 0xdeadbeef from
// the constructed peer, and the assertion fires.
//
// The doftcoin_2 regression shows the correct wiring: a state-var typed as the
// peer contract gets its address injected via constructor and cast back to the
// contract type, e.g. `vulnerableContract = Doftcoin(_vulnerableContract);`.
// Without that explicit injection the state var is dangling, as demonstrated
// below. This test is CORE FAILED on purpose — it pins the documented
// semantics, not an ESBMC bug.
pragma solidity >=0.8.0;

contract Inner {
    uint256 public sig;
    constructor() { sig = 0xdeadbeef; }
}

contract Outer {
    Inner public inner;
    function check() external {
        assert(inner.sig() == 0xdeadbeef);
    }
}
