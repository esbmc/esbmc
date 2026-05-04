// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract DeepJagged {
    // dynamic outer -> fixed[2] -> dynamic inner
    uint256[][2][] internal deep;

    function run() external {
        assert(deep.length == 0);

        deep.push();
        assert(deep.length == 1);

        deep[0][0].push(10);
        deep[0][1].push(20);

        assert(deep[0][0][0] == 10);
        assert(deep[0][1][0] == 20);
    }
}
