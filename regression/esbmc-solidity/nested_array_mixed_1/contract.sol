// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract MixedNestedPass {
    // fixed outer[2], dynamic middle, fixed inner[4]
    uint256[4][][2] internal mixed;

    function run() external {
        // outer fixed length
        assert(mixed.length == 2);

        // push a fixed[4] array to mixed[0]
        mixed[0].push();
        assert(mixed[0].length == 1);
        assert(mixed[0][0].length == 4);

        // write to fixed inner array elements
        mixed[0][0][0] = 42;
        mixed[0][0][1] = 43;
        mixed[0][0][2] = 44;
        mixed[0][0][3] = 45;
        assert(mixed[0][0][0] == 42);
        assert(mixed[0][0][3] == 45);

        // push another array to mixed[0]
        mixed[0].push();
        assert(mixed[0].length == 2);
        mixed[0][1][0] = 100;
        assert(mixed[0][1][0] == 100);

        // mixed[1] independent
        mixed[1].push();
        mixed[1][0][2] = 333;
        assert(mixed[1][0][2] == 333);

        // first value preserved
        assert(mixed[0][0][0] == 42);

        // overwrite and verify
        mixed[0][0][2] = 999;
        assert(mixed[0][0][2] == 999);

        // pop middle level
        mixed[0].pop();
        assert(mixed[0].length == 1);
        assert(mixed[0][0][0] == 42);
        assert(mixed[0][0][2] == 999);

        // push again after pop
        mixed[0].push();
        assert(mixed[0].length == 2);
        mixed[0][1][0] = 888;
        assert(mixed[0][1][0] == 888);
    }
}
