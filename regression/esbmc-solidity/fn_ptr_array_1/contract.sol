// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract Base {
    function _b(uint256 x) internal pure returns (uint256) {
        return x + 1;
    }
}

contract A is Base {
    function _self(uint256 x) internal pure returns (uint256) {
        return x + 10;
    }

    function dispatch(uint256 i, uint256 x) external pure returns (uint256) {
        function(uint256) internal pure returns (uint256)[3] memory ops =
            [_self, Base._b, _self];
        return ops[i](x);
    }
}
