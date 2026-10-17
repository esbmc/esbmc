// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// B8: inherited mapping addr-keyspace isolation between two derived
// instances.  Before the fix in solidity_convert_decl.cpp's
// move_to_initializer gate, an inherited mapping declared in Base was
// never per-instance-initialized in D's ctor — every D instance had
// m.addr=0 and writes aliased across instances.
contract Base {
    mapping(uint256 => uint256) internal m;
}

contract D is Base {
    function set(uint256 k, uint256 v) public { m[k] = v; }
    function get(uint256 k) public view returns (uint256) { return m[k]; }
}

contract H {
    function check(uint256 k, uint256 v1, uint256 v2) public {
        if (v1 == v2) return;
        D d1 = new D();
        D d2 = new D();
        d1.set(k, v1);
        d2.set(k, v2);
        // Each derived instance has its own mapping keyspace via
        // m.addr = this->$address set in D's ctor.
        assert(d1.get(k) == v1);
    }
}
