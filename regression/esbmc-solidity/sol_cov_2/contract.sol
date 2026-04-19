pragma solidity >=0.5.0;

contract Algo {
    address public proxy;

    constructor(address _proxy) {
        proxy = _proxy;
    }

    function Ass_1() public {
        require(msg.sender == proxy, "Not authorized: caller is not Proxy");
        assert(1 == 1);
    }

    function Ass_0() public {
        require(msg.sender == proxy, "Not authorized: caller is not Proxy");
        assert(0 == 1);
    }
}

contract Proxy {
    Algo public y;

    constructor() {
        address t = address(this);
        y = new Algo(t);
    }

    function test_op() public {
        y.Ass_0();
    }
}
