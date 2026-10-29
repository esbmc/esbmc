pragma solidity >=0.8.0;

contract Ext {
    mapping(address => uint256) public balances;

    function deposit() external payable {
        require(msg.value > 0);
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) external {
        require(amount > 0);
        uint256 balanceBefore = balances[msg.sender];

        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok);

        assert(balances[msg.sender] == balanceBefore);

        balances[msg.sender] -= amount;
    }
}
