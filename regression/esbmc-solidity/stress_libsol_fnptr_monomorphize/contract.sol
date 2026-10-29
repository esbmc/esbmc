// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

library L {
    function apply1(uint x, function(uint) pure returns (uint) f)
        internal pure returns (uint) { return f(x); }

    function applyLibCb(uint x) internal pure returns (uint) { return x + 10; }
}

function freeFnCb(uint x) pure returns (uint) { return x * 3; }

contract C {
    function ctor(uint x) internal pure returns (uint) { return x * x; }

    function check() public pure {
        assert(L.apply1(7, ctor) == 49);           // contract method cb
        assert(L.apply1(7, L.applyLibCb) == 17);   // library fn cb
        assert(L.apply1(7, freeFnCb) == 21);       // free fn cb
    }
}
