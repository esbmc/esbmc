// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract BB {
    constructor() {
        assert(1 == 2);
    }
}

contract DD {
    constructor() {
        BB x = new BB();
    }
}
