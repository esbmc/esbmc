// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract DoWhileTest {
    function sumUpTo(uint8 n) public pure returns (uint8) {
        uint8 sum = 0;
        uint8 i = 1;
        do {
            sum += i;
            i++;
        } while (i <= n);
        return sum;
    }

    function test() public pure {
        // sumUpTo(4) = 1 + 2 + 3 + 4 = 10
        assert(sumUpTo(4) == 10);
        // sumUpTo(1) = 1 (body executes at least once)
        assert(sumUpTo(1) == 1);
    }
}
