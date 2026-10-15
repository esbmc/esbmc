// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// L1.Data and L2.Data share the short name `Data`, and L2.Data contains
// L1.Data. When both were mapped to one tuple sort, that sort contained
// itself and z3 refused it ("datatype is not well-founded"). M1/M2 are the
// control: same shape, distinct short names.

library L1 {
    struct Data {
        uint256 _raw;
    }
}

library L2 {
    struct Data {
        L1.Data items;
    }
}

library M1 {
    struct AlphaData {
        uint256 _raw;
    }
}

library M2 {
    struct BetaData {
        M1.AlphaData items;
    }
}

contract D13_Z3TupleNotWellFounded {
    L2.Data private _collides;
    M2.BetaData private _distinct;

    function set(uint256 v) public {
        if (v > 5) {
            _collides.items._raw = v;
            _distinct.items._raw = v;
        }
        assert(_collides.items._raw == 0 || _collides.items._raw > 5);
        assert(_distinct.items._raw == _collides.items._raw);
    }
}
