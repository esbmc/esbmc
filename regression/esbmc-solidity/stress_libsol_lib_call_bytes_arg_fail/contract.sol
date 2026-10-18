// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual to stress_libsol_lib_call_bytes_arg_pass: validates that ESBMC
// can still find bugs through the library call argument coercion path.
// If the get_library_function_call coercion were silently dropped (the
// original bug), this contract would abort with a type-mismatch error
// instead of producing a usable VC, masking the assertion failure.

interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
}

library L {
    function _callOptionalReturn(IERC20 token, bytes memory data) private pure returns (uint256) {
        return data.length;
    }

    function safeTransfer(IERC20 token, address to, uint256 value) internal pure returns (uint256) {
        return _callOptionalReturn(
            token, abi.encodeWithSelector(token.transfer.selector, to, value));
    }
}

contract C {
    IERC20 t;
    function go(address to, uint256 v) public view {
        uint256 r = L.safeTransfer(t, to, v);
        // The library call must succeed (no type-mismatch abort) AND we
        // negate the property to force a counterexample, proving the
        // call boundary was actually exercised.
        assert(r == r + 1);
    }
}
