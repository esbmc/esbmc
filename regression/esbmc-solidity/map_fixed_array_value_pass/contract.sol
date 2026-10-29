// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: mapping(K => T[N]) — mapping whose value is a FIXED-size
// array.  Before the fix this tripped "unsupported mapping value type:
// sol_type=ARRAY_LITERAL" during frontend conversion.  The helper
// map_fixed_arr_get(&m, k, sz) lazily calloc's an N-slot slab on first
// access and returns the same pointer on subsequent reads, so writes
// persist across reads.
//
// The `new` path below puts Store in newContractSet so
// should_treat_as_new() returns true for its methods, exercising the
// map_fixed_arr_get call site in get_new_mapping_index_access (vs. the
// static index_exprt path used for simple static contracts).
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
    s.put(address(0x1), 0, 42);
    assert(s.get(address(0x1), 0) == 42);
  }
}
