// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    function test() public virtual {
        assert(1 == 1);
    }
}

contract B is Base {
    function test() public override {
        assert(1 == 0);
    }
    function test1(Base x, address _addr) public {
        x = new Base(); // x._ESBMC_cname = Base;
        x.test(); // if x._ESBMC_cname == x:
        // _ESBMC_Object_x.test();
        x = Base(_addr); // x._ESBMC_cname = x
        // if address == _ESBMC_Object_y.$address
        // x._ESBMC_cname = y;
        Base y = new Base();
        Base(_addr);
        new Base();
    }
}
