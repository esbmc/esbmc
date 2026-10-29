// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual of map_fixed_array_value_pass: distinct keys must not alias.
// If map_fixed_arr_get returned the same slab for different keys,
// the assertion below would hold and verification would pass — so a
// FAILED verdict confirms per-key allocation.
contract Store {
  mapping(address => uint256[3]) grid;

  function put(address who, uint256 i, uint256 v) public {
    grid[who][i] = v;
  }

  function get(address who, uint256 i) public view returns (uint256) {
    return grid[who][i];
  }
}

contract Harness {
  function test() public {
    Store s = new Store();
    s.put(address(0x1), 0, 7);
    assert(s.get(address(0x2), 0) == 7);
  }
}
