// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test diamond inheritance with virtual/override and explicit base calls.
// Final → Base1 → Emittable → Owned
//       → Base2 → Emittable → Owned
contract Owned {
    address payable owner;
    constructor() { owner = payable(msg.sender); }
}

contract Emittable is Owned {
    event Emitted();
    function emitEvent() virtual public {
        if (msg.sender == owner)
            emit Emitted();
    }
}

contract Base1 is Emittable {
    event Base1Emitted();
    function emitEvent() public virtual override {
        emit Base1Emitted();
        Emittable.emitEvent();
    }
}

contract Base2 is Emittable {
    event Base2Emitted();
    function emitEvent() public virtual override {
        emit Base2Emitted();
        Emittable.emitEvent();
    }
}

contract Final is Base1, Base2 {
    event FinalEmitted();
    function emitEvent() public override(Base1, Base2) {
        emit FinalEmitted();
        Base2.emitEvent();
    }
}
