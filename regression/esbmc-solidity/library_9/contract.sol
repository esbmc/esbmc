//SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library DataLib {
    struct Data {
        uint value;
    }

    function updateValue(Data storage data, uint newValue) public {
        data.value = newValue;
    }
}

contract UseLibrary {
    using DataLib for DataLib.Data;

    DataLib.Data internal myData;

    constructor() {
        myData.value = 100;
    }

    function changeValue(uint _val) public {
        myData.updateValue(_val);
    }

    function getValue() public view returns (uint) {
        return myData.value;
    }

    function testAssert() public view {
        assert(myData.value == 200);  
    }
}