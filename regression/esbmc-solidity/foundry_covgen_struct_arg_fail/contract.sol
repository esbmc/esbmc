// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
type Address is uint256;
type Timelocks is uint256;
struct Immutables {
    bytes32 orderHash;
    bytes32 hashlock;
    Address maker;
    Address taker;
    uint256 amount;
    Timelocks timelocks;
    bytes parameters;
}
contract IM {
    uint256 public s;
    function exec(bytes32 secret, Immutables calldata im) external {
        if (im.amount > 1000) { s = Address.unwrap(im.maker); } else { s = uint256(secret); }
    }
}
