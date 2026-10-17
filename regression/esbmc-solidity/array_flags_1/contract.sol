// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

// Dynamic array of fixed-size pairs: bool[2][].
// Push, set, resize via push/pop loops.
// From Solidity docs "Arrays" section (ArrayContract example).

contract ArrayContract {
    bool[2][] pairsOfFlags;

    function setFlagPair(uint index, bool flagA, bool flagB) public {
        pairsOfFlags[index][0] = flagA;
        pairsOfFlags[index][1] = flagB;
    }

    function changeFlagArraySize(uint newSize) public {
        if (newSize < pairsOfFlags.length) {
            while (pairsOfFlags.length > newSize)
                pairsOfFlags.pop();
        } else if (newSize > pairsOfFlags.length) {
            while (pairsOfFlags.length < newSize)
                pairsOfFlags.push();
        }
    }

    function addFlag(bool[2] memory flag) public returns (uint) {
        pairsOfFlags.push(flag);
        return pairsOfFlags.length;
    }

    function clear() public {
        delete pairsOfFlags;
    }
}
