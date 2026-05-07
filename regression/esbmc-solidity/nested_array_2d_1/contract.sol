// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract DynArray2DPass {
    uint256[][] internal a2;

    function run() external {
        assert(a2.length == 0);

        // push outer rows
        a2.push();
        a2.push();
        assert(a2.length == 2);
        assert(a2[0].length == 0);
        assert(a2[1].length == 0);

        // push values into inner arrays
        a2[0].push(100);
        a2[0].push(200);
        a2[0].push(300);
        assert(a2[0].length == 3);
        assert(a2[0][0] == 100);
        assert(a2[0][1] == 200);
        assert(a2[0][2] == 300);

        a2[1].push(999);
        assert(a2[1].length == 1);
        assert(a2[1][0] == 999);

        // rows are independent
        assert(a2[0][0] == 100);

        // pop from inner
        a2[0].pop();
        assert(a2[0].length == 2);
        assert(a2[0][1] == 200);

        // push to inner after pop
        a2[0].push(777);
        assert(a2[0].length == 3);
        assert(a2[0][2] == 777);

        // pop outer row
        a2.pop();
        assert(a2.length == 1);
        assert(a2[0][0] == 100);
    }
}
