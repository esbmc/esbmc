pragma solidity ^0.8.0;

struct T {
    bytes x;
    uint[3] y;
}
contract MyContract {
    bool public I;
    T public structJ;
}
contract tupleABITrig {
    MyContract public tupleABIContract;
}
