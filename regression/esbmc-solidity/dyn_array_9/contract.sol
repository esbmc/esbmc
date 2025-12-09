// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Return array of uint8 type
contract A {
    uint8[2] z; // int *z = calloc(2,1);

    
    function int8_call() public virtual  {


        z = [1, 2]; // int tmp3[2] = {0,1}
        z = [0];
        assert(z[1] == 2);
    }
}
