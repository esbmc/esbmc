// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for the goto-symex `function call: argument ... type
// mismatch: got unsignedbv, expected struct` abort on library-internal
// calls whose actual argument is `abi.encodeWithSelector(...)` (lowered
// to a uint256 identity in solidity_abi.c) and whose formal parameter is
// `bytes memory` (modelled as a BytesDynamic struct). The library call
// path (get_library_function_call) used to skip the formal/actual
// coercion, leaving a scalar argument bound to a struct parameter.
// SafeERC20._callOptionalReturn is the canonical real-world case.

interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
}

library L {
    function _callOptionalReturn(IERC20 token, bytes memory data) private {
        if (data.length > 0) {}
    }

    function safeTransfer(IERC20 token, address to, uint256 value) internal {
        _callOptionalReturn(
            token, abi.encodeWithSelector(token.transfer.selector, to, value));
    }
}

contract C {
    IERC20 t;
    function go(address to, uint256 v) public {
        L.safeTransfer(t, to, v);
        assert(true);
    }
}
