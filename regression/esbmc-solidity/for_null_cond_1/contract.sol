// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.6.0;

// contract.solast is solc 0.8.28 output with "condition": null added to the
// ForStatement: some solc JSON writers emit an omitted condition as null.
contract ForNullCond {
  function count() public pure {
    uint i;
    for (i = 0; ; i++) {
      if (i == 5) break;
    }
    assert(i == 5);
  }
}
