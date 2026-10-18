// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Whole-struct assignment into a mapping(address => Struct) via the
// `new` path, which routes through map_generic_set (the library helper
// for struct-valued mappings under is_new_expr=true).
//
// Before the fix, map_generic_set received the value symbol in its
// size_t slot instead of sizeof(value), which worked by accident for
// small structs but errored "got struct, expected unsignedbv" as soon
// as the struct grew past a few fields (first hit on SolidiFi buggy_24's
// 18-field Record).
//
// Harness instantiates Store via `new Store()` to force should_treat_as_new
// → the is_new_expr branch in get_new_mapping_index_access emits
// map_generic_set with the sizeof(value) arg, which our fix now supplies.

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

  function getA(address who) public view returns (uint256) {
    return users[who].a;
  }

  function getB(address who) public view returns (uint256) {
    return users[who].b;
  }
}

contract Harness {
  function test() public {
    Store s = new Store();
    s.set(address(0x1), 10);
    assert(s.getA(address(0x1)) + 1 == s.getB(address(0x1)));
  }
}
