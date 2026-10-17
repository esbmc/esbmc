// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Completeness pin (KNOWNBUG, FAIL variant): same nested-push
// shape as the PASS pin with a planted bug at arr[0][base+1].
// Should falsify under base-case BMC, BUT current state
// (2026-05-10) is the same glibc-malloc crash during symex —
// the per-byte memcpy preservation chain in _ESBMC_array_push
// breaks before the SAT solver even sees the formula.
//
// Pins the bug from both PASS-side and FAIL-side. A model fix
// that resolves the crash should flip BOTH variants in the same
// commit.
contract C {
    bytes32[][] internal arr;

    function run() external {
        if (arr.length == 0)
            arr.push();
        uint256 base = arr[0].length;
        arr[0].push(bytes32(uint256(0x01)));
        arr[0].push(bytes32(uint256(0x02)));
        arr[0].push(bytes32(uint256(0x03)));
        // FLIPPED: arr[0][base+1] is 0x02, not 0x99
        assert(arr[0][base + 1] == bytes32(uint256(0x99)));
    }
}
