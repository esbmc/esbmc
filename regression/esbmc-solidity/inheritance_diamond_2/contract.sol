// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test multi-level inheritance with constructor args and --bound mode.
// PriceFeed → Named("GoldFeed") → Owned
//           → Emittable → Owned
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

contract Named is Owned, Emittable {
    bytes32 public name;
    constructor(bytes32 _name) {
        name = _name;
    }
    function emitEvent() public virtual override {
        if (msg.sender == owner) {
            Emittable.emitEvent();
        }
    }
}

contract PriceFeed is Owned, Emittable, Named("GoldFeed") {
    uint info;

    function updateInfo(uint newInfo) public {
        if (msg.sender == owner) info = newInfo;
    }

    function emitEvent() public override(Emittable, Named) { Named.emitEvent(); }
    function get() public view returns(uint r) { return info; }
}
