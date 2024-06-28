// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// 定义子合约 SubContract
contract SubContract {
    uint256 public value;

    constructor(uint256 _value) {
        value = _value;
    }

    function setValue(uint256 _value) public {
        value = _value;
    }
}

// 定义主合约 MainContract
contract MainContract {
    SubContract public subContract;

    constructor(uint256 _initialValue) {
        // 实例化子合约并传递 _initialValue
        subContract = new SubContract(_initialValue);
    }

    function setSubContractValue(uint256 _value) public {
        subContract.setValue(_value);
    }

    function getSubContractValue() public view returns (uint256) {
        return subContract.value();
    }
}
