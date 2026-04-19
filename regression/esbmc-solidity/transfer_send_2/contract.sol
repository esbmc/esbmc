pragma solidity ^0.8.0;

// Simple token with a boolean‐returning transfer
contract SimpleToken {
    mapping(address => uint256) public balances;

    constructor() {
        // Give deployer 100 tokens
        balances[msg.sender] = 100;
    }

    /// @notice Transfer tokens, returning false on failure
    function transfer(address to, uint256 amount) public returns (bool) {
        if (balances[msg.sender] < amount) {
            return false;
        }
        balances[msg.sender] -= amount;
        balances[to] += amount;
        return true;
    }
}

// Test contract that will trigger an assertion failure
contract TestTransfer {
    function test() public {
        // Deploy a fresh SimpleToken
        SimpleToken token = new SimpleToken();

        // Attempt to transfer more tokens than exist (100)
        bool ok = token.transfer(address(this), 200);

        // Use the return value of transfer – this will be false,
        // so the following assert will fail under ESBMC.
        assert(ok);
    }
}
