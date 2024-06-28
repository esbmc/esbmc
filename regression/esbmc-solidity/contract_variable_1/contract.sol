// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ExistingContract {
    function setValue(uint256 _value) public{}
    function getValue() public view returns (uint256){}
}

contract MainContract is ExistingContract  {
    ExistingContract public existingContract;

    constructor(address _contractAddress) {
        existingContract = ExistingContract(_contractAddress);
    }

    function setExistingContractValue(uint256 _value) public {
        existingContract.setValue(_value);
    }

    function getExistingContractValue() public view returns (uint256) {
        return existingContract.getValue();
    }
}

