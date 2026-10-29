// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Library + user-defined value type (UDVT) coverage-test generation. A library
// has no dispatcher, so the generator's --function isolated-function path
// reconstructs a static call `TraitsLib.hasFlag(Traits.wrap(N))` with the exact
// UDVT wrapper (a bare uint256 is not assignable to a Traits parameter) and no
// setUp instance (a library is never `new`'d). Both sides of the if-decision
// are covered, so two cases are generated.
type Traits is uint256;

library TraitsLib {
    uint256 private constant _FLAG = 1 << 249;
    function hasFlag(Traits t) internal pure returns (bool) {
        if ((Traits.unwrap(t) & _FLAG) != 0)
            return true;
        return false;
    }
}
