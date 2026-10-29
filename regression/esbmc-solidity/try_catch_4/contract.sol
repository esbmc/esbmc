// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.1;

interface Oracle {
    function getPrice() external returns (uint);
}

// Tests try/catch failure: assertion only holds on success path
contract OracleConsumer {
    Oracle oracle;
    uint public price;
    bool public updated;

    function fetchPrice() public {
        try oracle.getPrice() returns (uint p) {
            price = p;
            updated = true;
        } catch {
            updated = false;
        }
        // FAIL: updated can be false in catch branch
        assert(updated == true);
    }
}
