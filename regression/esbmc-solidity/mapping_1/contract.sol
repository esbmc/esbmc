// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    enum T {
        xxx
    }
    mapping(int8 => uint) test2;

    function test() public {
        int8 x = 0;
        assert(test2[0] == 0);
        assert(test2[x] == 0);
		assert(test2[-1] == 0);
		assert(test2[1] == 0);
    }
}
