// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Inheritance — Derived inherits both a base storage var and a base
// public setter.  After cloning Derived, the base-contract slot must
// be copied just like Derived's own slot.  Probes whether the
// whole-struct copy reaches into the merged-struct layout that ESBMC
// produces for inherited contracts.
function __ESOL_deep_copy(Derived src) pure returns (Derived) { return src; }

contract Base {
    uint256 public bx;
    function setB(uint256 v) public { bx = v; }
}

contract Derived is Base {
    uint256 public dx;
    function setD(uint256 v) public { dx = v; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        Derived base = new Derived();
        base.setB(a);
        base.setD(b);
        Derived clone = __ESOL_deep_copy(base);
        assert(clone.bx() == a);
        assert(clone.dx() == b);
    }
}
