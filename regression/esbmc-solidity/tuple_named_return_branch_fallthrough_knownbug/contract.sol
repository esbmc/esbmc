// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// KNOWNBUG — pre-existing (NOT introduced by the tuple-binding fix): a tuple
// function mixing an early explicit `return (x,y)` in one branch with a
// named-return fall-through in another. The GOTO lowering drops the
// post-branch `a=3; b=4;` statements and mangles the branch return
// (`if(c) return(1,2)` collapses so both paths yield (1,2)); the fall-through
// (c==false) path never gets (3,4). This is a back-block/return lowering gap
// independent of tuple named-return binding. The aqua/BalanceLib targets do not
// mix branch-return with named fall-through. Pinned until the lowering is fixed.
contract H {
    function f(bool c) public pure returns (uint a, uint b) {
        if (c) return (1, 2);
        a = 3;
        b = 4;
    }
    function check(bool c) public {
        (uint ra, uint rb) = f(c);
        if (c) {
            assert(ra == 1 && rb == 2);
        } else {
            assert(ra == 3 && rb == 4);
        }
    }
}
