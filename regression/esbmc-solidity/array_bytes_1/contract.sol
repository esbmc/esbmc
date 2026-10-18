// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

// Bytes array operations: assignment, push, index write, delete element.
// From Solidity docs "Arrays" section (ArrayContract.byteArrays).

contract ArrayContract {
    bytes byteData;

    function byteArrays(bytes memory data) public {
        byteData = data;
        for (uint i = 0; i < 7; i++)
            byteData.push();
        byteData[3] = 0x08;
        delete byteData[2];
    }
}
// KNOWNBUG: upstream k-induction disables the inductive step here ("cannot
// havoc what a loop writes through a pointer"), so only base cases run and
// the expected proof is never reached.
