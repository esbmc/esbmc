// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

// Struct-in-mapping with direct field writes (no storage ref).
// From Solidity docs "Structs" section (CrowdFunding example).

struct Funder {
    address addr;
    uint amount;
}

contract CrowdFunding {
    struct Campaign {
        address payable beneficiary;
        uint fundingGoal;
        uint numFunders;
        uint amount;
    }

    uint numCampaigns;
    mapping(uint => Campaign) campaigns;
    mapping(uint => mapping(uint => Funder)) campaignFunders;

    function test_direct_write() public {
        campaigns[0].beneficiary = payable(address(0x1234));
        campaigns[0].fundingGoal = 100;
        campaigns[0].numFunders = 0;
        campaigns[0].amount = 0;
        numCampaigns = 1;

        assert(numCampaigns == 1);
        assert(campaigns[0].fundingGoal == 100);
        assert(campaigns[0].amount == 0);

        campaigns[0].amount += 50;
        assert(campaigns[0].amount == 50);

        campaignFunders[0][0] = Funder({addr: msg.sender, amount: 50});
        campaigns[0].numFunders = 1;
        assert(campaigns[0].numFunders == 1);
    }
}
