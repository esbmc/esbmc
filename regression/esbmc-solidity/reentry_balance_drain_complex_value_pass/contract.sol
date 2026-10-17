pragma solidity >=0.8.0;
// Exercises the single-evaluation guarantee: the value expression
// computeAmount() must evaluate exactly once across the call argument
// and the post-assert that references the same snapshot.  CEI ordering
// keeps the property satisfied under reentry.
contract Vault {
    mapping(address => uint256) public credit;
    uint256 sentinel;
    function deposit() external payable { credit[msg.sender] += msg.value; }
    function computeAmount() internal returns (uint256) {
        sentinel++;
        return credit[msg.sender];
    }
    function withdraw() external {
        uint256 amt = computeAmount();
        require(amt > 0);
        credit[msg.sender] = 0;
        payable(msg.sender).transfer(amt);
    }
}
contract Attacker {
    Vault target;
    bool hit;
    constructor(Vault _t) { target = _t; }
    function attack() external { target.withdraw(); }
    receive() external payable {
        if (!hit && address(target).balance >= msg.value) {
            hit = true;
            target.withdraw();
        }
    }
}
