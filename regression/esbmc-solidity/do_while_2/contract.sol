// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract DoWhileTest {
    function test() public pure {
        uint8 counter = 0;
        // do-while always executes body at least once
        do {
            counter++;
        } while (false);
        // counter must be 1, not 0
        assert(counter == 0);  // should FAIL: counter is 1
    }
}
