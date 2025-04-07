// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Defining contract A
contract A {
    // Declaring internal
    // state variable
    string internal x;

    // Defining external function
    // to set value of
    // internal state variable x
    function setA() external {
        x = "test";
    }
}

// Defining contract B
contract B is A {
    // Declaring internal
    // state variable
    uint internal pow;

    // Defining external function
    // to set value of internal
    // state variable pow
    function setB() external {
        uint a = 2;
        uint b = 20;
        pow = a ** b;
    }
}

// Defining child contract C
// inheriting parent contract
// A and B
contract C is B {
    // Defining external function
    // to return state variable x
    function getStr() external returns (string memory) {
        B xx = new B();
        return x;
    }

    // Defining external function
    // to return state variable pow
    function getPow() external returns (uint) {
        return pow;
    }
}

// Defining calling contract
contract caller {
    // Creating object of contract C
    C contractC = new C();

    // Defining public function to
    // return values from functions
    // getStr and getPow
    function testInheritance() public returns (string memory) {
        contractC.setA();
        contractC.setB();
        return (contractC.getStr());
    }
}
