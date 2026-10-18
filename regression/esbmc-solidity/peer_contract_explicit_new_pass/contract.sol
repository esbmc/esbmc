// Worked example #1 documenting --bound semantics for peer contract instances.
//
// Pair with peer_contract_state_var_dangling_fail: the only difference is HOW
// the Inner reference is acquired. Here Outer creates the instance explicitly
// via `new Inner()`, which runs Inner's constructor and sets sig = 0xdeadbeef.
// The companion test declares `Inner public inner;` as a state variable and
// never assigns it; the dangling reference does NOT pick up Inner's constructed
// state, so the assertion there fails. Together the pair shows that `--bound`
// binds addresses on _ESBMC_Object_<C> peer instances but does not auto-wire
// uninitialised user-declared state variables to those instances — matching
// real Solidity semantics where state-var refs default to address(0).
pragma solidity >=0.8.0;

contract Inner {
    uint256 public sig;
    constructor() { sig = 0xdeadbeef; }
}

contract Outer {
    function check() external {
        Inner inner = new Inner();
        assert(inner.sig() == 0xdeadbeef);
    }
}
