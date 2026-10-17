// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// TOD race on an internal (default-visibility) state variable.
// Mirrors SolidiFi's injected pattern: `setReward` and `claimReward`
// both read/write `claimed_TODn`, but `claimed_TODn` is declared with
// no explicit visibility → default internal in solc ≥0.5.
//
// The TOD race harness must still detect the race even though there
// is no public getter — ESBMC injects shadow public getters
// (`__tod_get_claimed`, `__tod_get_r`) into the extracted contract
// source so the two copies' post-call states are comparable.
contract C {
    uint256 internal r;
    bool private claimed;

    function setR(uint256 v) public {
        require(!claimed);
        r = v;
    }

    function claimR() public {
        require(!claimed);
        claimed = true;
    }
}
