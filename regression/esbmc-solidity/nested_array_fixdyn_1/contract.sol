// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract FixedDynPass {
    uint256[][3] internal a3;

    function run() external {
        // outer is fixed length 3
        assert(a3.length == 3);

        // inner arrays start empty
        assert(a3[0].length == 0);
        assert(a3[1].length == 0);
        assert(a3[2].length == 0);

        // push to different inner arrays
        a3[0].push(10);
        a3[0].push(20);
        a3[1].push(100);
        a3[2].push(200);
        a3[2].push(300);
        a3[2].push(400);

        assert(a3[0].length == 2);
        assert(a3[1].length == 1);
        assert(a3[2].length == 3);

        assert(a3[0][0] == 10);
        assert(a3[0][1] == 20);
        assert(a3[1][0] == 100);
        assert(a3[2][0] == 200);
        assert(a3[2][1] == 300);
        assert(a3[2][2] == 400);

        // pop from inner
        a3[2].pop();
        assert(a3[2].length == 2);

        // push after pop
        a3[2].push(500);
        assert(a3[2].length == 3);
        assert(a3[2][2] == 500);

        // other arrays unaffected
        assert(a3[0][0] == 10);
        assert(a3[1][0] == 100);
    }
}
