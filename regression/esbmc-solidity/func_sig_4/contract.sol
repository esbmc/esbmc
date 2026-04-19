// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract A {
    uint public val = 1;
    function test() public {
        assert(1 == 1);
    }
}

contract B {
    int public val = 2;
    function test() external {
        assert(1 == 0);
    }
}

contract C {
    function test(address a1, address a2) public {
        A a = A(a1);
        A b = A(a2);

        // this block the possibility of b == B
        assert(a.val() + b.val() == 2); // 1 + 2 = 3

        a.test(); // OK: A.test is public
        b.test(); // OK: B.test is external, but called as ABI external call
    }
}
