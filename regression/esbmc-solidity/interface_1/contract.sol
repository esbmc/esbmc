pragma solidity >=0.5.0;

interface caculator {
    function getResult() external view returns (int8);
}

contract Test is caculator{
    constructor() {}
    function getResult() external view returns (int8) {
        int8 a = 127;
        int8 b = 0;
        int8 result = a / b;
        return result;
    }
}
