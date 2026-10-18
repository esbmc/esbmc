// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Aqua's revert form: require(cond, CustomErrorInstance) (Solidity >=0.8.26).
// The frontend drops the error arg and lowers via the require rollback path, so
// there is NO detectable revert terminator (unlike `revert E()`). The generator
// must therefore wrap the covering call in a revert-tolerant try/catch so the
// assertion-free replay stays a PASS in forge instead of aborting on the revert.
contract Req {
    uint256 public x;
    error Denied(uint256 v);

    function poke(uint256 v) external {
        require(v != 7, Denied(v));
        x = v;
    }
}
