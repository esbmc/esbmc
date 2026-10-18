// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// A fixed-size array struct field cannot be rendered faithfully as a positional
// literal (a dynamic `new T[](N)` default is illegal for a fixed field), so the
// whole struct degrades to UNSUPPORTED (all-or-nothing; never a wrong literal).
struct Box {
    uint256 tag;
    uint256[2] cells;
}
contract BoxUser {
    uint256 public s;
    function put(Box calldata b) external {
        if (b.tag > 7) { s = b.tag; } else { s = 0; }
    }
}
