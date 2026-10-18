// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual of map_large_struct_set_pass. Same setter path exercised from
// Harness.test so the frontend still emits map_generic_set with the
// multi-field value; assertion is placed before the call so verification
// reaches a reachable `assert(false)` regardless of the mapping's
// get/set round-trip semantics.

contract Store {
  struct User {
    uint256 a;
    uint256 b;
    uint256 c;
    uint256 d;
    uint256 e;
    uint256 f;
    uint256 g;
    uint256 h;
    bool active;
  }

  mapping(address => User) public users;

  function set(address who, uint256 v) public {
    users[who] = User(v, v + 1, v + 2, v + 3, v + 4, v + 5, v + 6, v + 7, true);
  }
}

contract Harness {
  function test() public {
    Store s = new Store();
    assert(false);
    s.set(address(0x1), 10);
  }
}
