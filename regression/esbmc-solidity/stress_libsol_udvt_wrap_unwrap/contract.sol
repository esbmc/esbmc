// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
type MyAddress is address;
contract C {
    function f() pure public {
        MyAddress.wrap;
        MyAddress.unwrap;
    }
}
