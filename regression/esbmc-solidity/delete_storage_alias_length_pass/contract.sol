// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Solidity spec example for `delete`: a storage reference aliasing the
// state variable must observe the length reset after the original is
// deleted.
//
// Bug A downstream: y aliases dataArray, both read the same
// dataArray_dynarray_len[$address] slot. Currently fails. KNOWNBUG.
contract C {
    uint[] dataArray;

    function f() public {
        require(dataArray.length == 0);
        dataArray.push(1);
        dataArray.push(2);
        uint[] storage y = dataArray;
        delete dataArray;
        assert(y.length == 0);
    }
}
