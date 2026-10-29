// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

// Storage reference write to mapping element: writes through
// `Campaign storage c = campaigns[0]; c.field = val;` should
// propagate back to the mapping.
// From Solidity docs "Structs" section (CrowdFunding pattern).

contract CrowdFunding {
    struct Campaign {
        uint fundingGoal;
        uint amount;
    }

    mapping(uint => Campaign) campaigns;

    function test_ref_write() public {
        Campaign storage c = campaigns[0];
        c.fundingGoal = 100;
        c.amount = 50;

        assert(campaigns[0].fundingGoal == 100);
        assert(campaigns[0].amount == 50);
    }
}
