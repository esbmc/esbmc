// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

interface IB {
    struct Order {
        uint256 x;
    }
}

library L {
    function build() internal pure returns (IB.Order memory order) {
        order.x = 42;
    }
}

contract A {
    function f() external pure returns (uint256) {
        IB.Order memory o = L.build();
        return o.x;
    }
}
