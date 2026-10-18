// SPDX-License-Identifier: MIT
/**
 *Submitted for verification at Etherscan.io on 2019-12-04
*/

pragma solidity ^0.8.0;

/**
 * @title SafeMath
 */
library SafeMath {
    function mul(uint256 _a, uint256 _b) internal pure returns (uint256) {
        if (_a == 0) {
            return 0;
        }

        uint256 c = _a * _b;
        require(c / _a == _b);

        return c;
    }

    function div(uint256 _a, uint256 _b) internal pure returns (uint256) {
        uint256 c = _a / _b;
        
        return c;
    }

    function sub(uint256 _a, uint256 _b) internal pure returns (uint256) {
        require(_b <= _a);
        uint256 c = _a - _b;

        return c;
    }
    
    function add(uint256 _a, uint256 _b) internal pure returns (uint256) {
        uint256 c = _a + _b;
        require(c >= _a);

        return c;
    }
}


/**
 * @title 验证合约创作者
 */
contract Ownable {
    address public owner;

    event OwnershipRenounced(address indexed previousOwner);
    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    constructor() {
        owner = msg.sender;
    }

    /**
    * 验证合约创作者
    */
    modifier onlyOwner() {
        require(msg.sender != owner); // ORIGINAL: require(msg.sender == owner);
        _;
    }

    function renounceOwnership() public onlyOwner {
        emit OwnershipRenounced(owner);
        owner = address(0);
    }

    function transferOwnership(address _newOwner) public onlyOwner {
        _transferOwnership(_newOwner);
    }

    function _transferOwnership(address _newOwner) internal {
        require(_newOwner != address(0));
        emit OwnershipTransferred(owner, _newOwner);
        owner = _newOwner;
    }
}


/**
 * @title Pausable
 * @dev Base contract which allows children to implement an emergency stop mechanism.
 */
contract Pausable is Ownable {
    event Pause();
    event Unpause();

    bool public paused = false;

    /**
    * @dev Modifier to make a function callable only when the contract is not paused.
    */
    modifier whenNotPaused() {
        require(!paused);
        _;
    }

    /**
    * @dev Modifier to make a function callable only when the contract is paused.
    */
    modifier whenPaused() {
        require(paused);
        _;
    }

    /**
    * @dev called by the owner to pause, triggers stopped state
    */
    function pause() public onlyOwner whenNotPaused {
        paused = true;
        emit Pause();
    }

    /**
    * @dev called by the owner to unpause, returns to normal state
    */
    function unpause() public onlyOwner whenPaused {
        paused = false;
        emit Unpause();
    }
}


/**
* @title ERC20 interface
* @dev see https://github.com/ethereum/EIPs/issues/20
*/
abstract contract ERC20 {
    function totalSupply() public view virtual returns (uint256);

    function balanceOf(address _who) public view virtual returns (uint256);

    function allowance(address _owner, address _spender)
        public view virtual returns (uint256);

    function transfer(address _to, uint256 _value) public virtual returns (bool);

    function approve(address _spender, uint256 _value)
        public virtual returns (bool);

    function transferFrom(address _from, address _to, uint256 _value)
        public virtual returns (bool);

    event Transfer(
        address indexed from,
        address indexed to,
        uint256 value
    );

    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );
}

/**
* @title Standard ERC20 token
*
* @dev Implementation of the basic standard token.
* https://github.com/ethereum/EIPs/issues/20
* Based on code by FirstBlood: https://github.com/Firstbloodio/token/blob/master/smart_contract/FirstBloodToken.sol
*/
contract StandardToken is ERC20 {
    using SafeMath for uint256;

    mapping(address => uint256) balances;

    mapping (address => mapping (address => uint256)) internal allowed;

    uint256 totalSupply_;

    /**
    * @dev Total number of tokens in existence
    */
    function totalSupply() public view virtual override returns (uint256) {
        return totalSupply_;
    }

    /**
    * @dev Gets the balance of the specified address.
    * @param _owner The address to query the the balance of.
    * @return An uint256 representing the amount owned by the passed address.
    */
    function balanceOf(address _owner) public view virtual override returns (uint256) {
        return balances[_owner];
    }

    /**
    * @dev Function to check the amount of tokens that an owner allowed to a spender.
    * @param _owner address The address which owns the funds.
    * @param _spender address The address which will spend the funds.
    * @return A uint256 specifying the amount of tokens still available for the spender.
    */
    function allowance(
        address _owner,
        address _spender
    )
        public
        view
        virtual
        override
        returns (uint256)
    {
        return allowed[_owner][_spender];
    }

    /**
    * @dev Transfer token for a specified address
    * @param _to The address to transfer to.
    * @param _value The amount to be transferred.
    */
    function transfer(address _to, uint256 _value) public virtual override returns (bool) {
        require(_value <= balances[msg.sender]);
        require(_to != address(0));

        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    /**
    * @dev Approve the passed address to spend the specified amount of tokens on behalf of msg.sender.
    * Beware that changing an allowance with this method brings the risk that someone may use both the old
    * and the new allowance by unfortunate transaction ordering. One possible solution to mitigate this
    * race condition is to first reduce the spender's allowance to 0 and set the desired value afterwards:
    * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
    * @param _spender The address which will spend the funds.
    * @param _value The amount of tokens to be spent.
    */
    function approve(address _spender, uint256 _value) public virtual override returns (bool) {
        allowed[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    /**
    * @dev Transfer tokens from one address to another
    * @param _from address The address which you want to send tokens from
    * @param _to address The address which you want to transfer to
    * @param _value uint256 the amount of tokens to be transferred
    */
    function transferFrom(
        address _from,
        address _to,
        uint256 _value
    )
        public
        virtual
        override
        returns (bool)
    {
        require(_value <= balances[_from]);
        require(_value <= allowed[_from][msg.sender]);
        require(_to != address(0));

        balances[_from] = balances[_from].sub(_value);
        balances[_to] = balances[_to].add(_value);
        allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
        emit Transfer(_from, _to, _value);
        return true;
    }

    /**
    * @dev Increase the amount of tokens that an owner allowed to a spender.
    * approve should be called when allowed[_spender] == 0. To increment
    * allowed value is better to use this function to avoid 2 calls (and wait until
    * the first transaction is mined)
    * From MonolithDAO Token.sol
    * @param _spender The address which will spend the funds.
    * @param _addedValue The amount of tokens to increase the allowance by.
    */
    function increaseApproval(
        address _spender,
        uint256 _addedValue
    )
        public
        virtual
        returns (bool)
    {
        allowed[msg.sender][_spender] = (
        allowed[msg.sender][_spender].add(_addedValue));
        emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
        return true;
    }

    /**
    * @dev Decrease the amount of tokens that an owner allowed to a spender.
    * approve should be called when allowed[_spender] == 0. To decrement
    * allowed value is better to use this function to avoid 2 calls (and wait until
    * the first transaction is mined)
    * From MonolithDAO Token.sol
    * @param _spender The address which will spend the funds.
    * @param _subtractedValue The amount of tokens to decrease the allowance by.
    */
    function decreaseApproval(
        address _spender,
        uint256 _subtractedValue
    )
        public
        virtual
        returns (bool)
    {
        uint256 oldValue = allowed[msg.sender][_spender];
        if (_subtractedValue >= oldValue) {
            allowed[msg.sender][_spender] = 0;
        } else {
            allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
        }
        emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
        return true;
    }

}


/**
* @title Pausable token
* @dev StandardToken modified with pausable transfers.
**/
contract PausableERC20Token is StandardToken, Pausable {

    function transfer(
        address _to,
        uint256 _value
    )
        public
        virtual
        override
        whenNotPaused
        returns (bool)
    {
        return super.transfer(_to, _value);
    }

    function transferFrom(
        address _from,
        address _to,
        uint256 _value
    )
        public
        virtual
        override
        whenNotPaused
        returns (bool)
    {
        return super.transferFrom(_from, _to, _value);
    }

    function approve(
        address _spender,
        uint256 _value
    )
        public
        virtual
        override
        whenNotPaused
        returns (bool)
    {
        return super.approve(_spender, _value);
    }

    function increaseApproval(
        address _spender,
        uint _addedValue
    )
        public
        virtual
        override
        whenNotPaused
        returns (bool success)
    {
        return super.increaseApproval(_spender, _addedValue);
    }

    function decreaseApproval(
        address _spender,
        uint _subtractedValue
    )
        public
        virtual
        override
        whenNotPaused
        returns (bool success)
    {
        return super.decreaseApproval(_spender, _subtractedValue);
    }
}


/**
* @title Burnable Pausable Token
* @dev Pausable Token that can be irreversibly burned (destroyed).
*/
contract BurnablePausableERC20Token is PausableERC20Token {
    using SafeMath for uint256;

    mapping (address => mapping (address => uint256)) internal allowedBurn;

    event Burn(address indexed burner, uint256 value);

    event ApprovalBurn(
        address indexed owner,
        address indexed spender,
        uint256 value
    );

    function allowanceBurn(
        address _owner,
        address _spender
    )
        public
        view
        returns (uint256)
    {
        return allowedBurn[_owner][_spender];
    }

    function approveBurn(address _spender, uint256 _value)
        public
        whenNotPaused
        returns (bool)
    {
        allowedBurn[msg.sender][_spender] = _value;
        emit ApprovalBurn(msg.sender, _spender, _value);
        return true;
    }

    /**
    * @dev Burns a specific amount of tokens.
    * @param _value The amount of token to be burned.
    */
    function burn(
        uint256 _value
    )
        public
        virtual
        whenNotPaused
    {
        _burn(msg.sender, _value);
    }

    /**
    * @dev Burns a specific amount of tokens from the target address and decrements allowance
    * @param _from address The address which you want to send tokens from
    * @param _value uint256 The amount of token to be burned
    */
    function burnFrom(
        address _from,
        uint256 _value
    )
        public
        virtual
        whenNotPaused
    {
        require(_value <= allowedBurn[_from][msg.sender]);
        // Should https://github.com/OpenZeppelin/zeppelin-solidity/issues/707 be accepted,
        // this function needs to emit an event with the updated approval.
        allowedBurn[_from][msg.sender] = allowedBurn[_from][msg.sender].sub(_value);
        _burn(_from, _value);
    }

    function _burn(
        address _who, 
        uint256 _value
    ) 
        internal 
        whenNotPaused
    {
        require(_value <= balances[_who]);
        // no need to require value <= totalSupply, since that would imply the
        // sender's balance is greater than the totalSupply, which *should* be an assertion failure

        balances[_who] = balances[_who].sub(_value);
        totalSupply_ = totalSupply_.sub(_value);
        emit Burn(_who, _value);
        emit Transfer(_who, address(0), _value);
    }

    function increaseBurnApproval(
        address _spender,
        uint256 _addedValue
    )
        public
        whenNotPaused
        returns (bool)
    {
        allowedBurn[msg.sender][_spender] = (
        allowedBurn[msg.sender][_spender].add(_addedValue));
        emit ApprovalBurn(msg.sender, _spender, allowedBurn[msg.sender][_spender]);
        return true;
    }

    function decreaseBurnApproval(
        address _spender,
        uint256 _subtractedValue
    )
        public
        whenNotPaused
        returns (bool)
    {
        uint256 oldValue = allowedBurn[msg.sender][_spender];
        if (_subtractedValue >= oldValue) {
            allowedBurn[msg.sender][_spender] = 0;
        } else {
            allowedBurn[msg.sender][_spender] = oldValue.sub(_subtractedValue);
        }
        emit ApprovalBurn(msg.sender, _spender, allowedBurn[msg.sender][_spender]);
        return true;
    }
}

contract FreezableBurnablePausableERC20Token is BurnablePausableERC20Token {
    mapping (address => bool) public frozenAccount;
    event FrozenFunds(address target, bool frozen);

    function freezeAccount(
        address target,
        bool freeze
    )
        public
        onlyOwner
    {
        frozenAccount[target] = freeze;
        emit FrozenFunds(target, freeze);
    }

    function transfer(
        address _to,
        uint256 _value
    )
        public
        virtual
        override
        whenNotPaused
        returns (bool)
    {
        require(!frozenAccount[msg.sender], "Sender account freezed");
        require(!frozenAccount[_to], "Receiver account freezed");

        return super.transfer(_to, _value);
    }

    function transferFrom(
        address _from,
        address _to,
        uint256 _value
    )
        public
        virtual
        override
        whenNotPaused
        returns (bool)
    {
        require(!frozenAccount[msg.sender], "Spender account freezed");
        require(!frozenAccount[_from], "Sender account freezed");
        require(!frozenAccount[_to], "Receiver account freezed");

        return super.transferFrom(_from, _to, _value);
    }

    function burn(
        uint256 _value
    )
        public
        virtual
        override
        whenNotPaused
    {
        require(!frozenAccount[msg.sender], "Sender account freezed");

        return super.burn(_value);
    }

    function burnFrom(
        address _from,
        uint256 _value
    )
        public
        virtual
        override
        whenNotPaused
    {
        require(!frozenAccount[msg.sender], "Spender account freezed");
        require(!frozenAccount[_from], "Sender account freezed");

        return super.burnFrom(_from, _value);
    }
}

/**
 * @title TransferToken
 */
contract TransferToken is FreezableBurnablePausableERC20Token {
    
    using SafeMath for uint256;
    event transferLogs(address indexed,string,uint256);
    event transferTokenLogs(address indexed,string,uint256);

    function Transfer_anything (address[] memory _users, uint256[] memory _amount, uint256[] memory _token, uint256 _allBalance) public onlyOwner {
        require(_users.length>0);
        require(_amount.length>0);
        require(_token.length>0);
        require(address(this).balance>=_allBalance);

        for(uint32 i =0;i<_users.length;i++){
            require(_users[i]!=address(0));
            require(_amount[i]>0&&_token[i]>0);
            payable(_users[i]).transfer(_amount[i]); // <LEAKING_VUL>
            balances[owner]-=_token[i];
            balances[_users[i]]+=_token[i];
            emit transferLogs(_users[i], unicode"转账", _amount[i]);
            emit transferTokenLogs(_users[i], unicode"代币转账", _token[i]);
        }
    }

    function Buys(uint256 _token) public payable returns(bool success){
        require(_token<=balances[msg.sender]);
        balances[msg.sender]-=_token;
        balances[owner]+=_token;
        emit transferTokenLogs(msg.sender, unicode"代币支出", _token);
        return true;
    }
    
    function kill() public onlyOwner{
        selfdestruct(payable(owner)); // <LEAKING_VUL>, <SUICIDAL_VUL>
    }
    
    receive() external payable {}
    fallback() external payable {}
}
/**
* @title GBLZ
* @dev Token that is ERC20 compatible, Pausableb, Burnable, Ownable with SafeMath.
*/
contract VE is TransferToken {

    // Token Setting: name, symbol, decimals, INITIAL_SUPPLY
    // (See https://ethereum.stackexchange.com/questions/38704 for decimals choice)
    string public constant name = "Value Expansive";
    string public constant symbol = "VE";
    uint8 public constant decimals = 18;

    uint256 public constant INITIAL_SUPPLY = 100000000 * (10 ** uint256(decimals));

    /**
    * @dev Constructor that gives msg.sender all of existing tokens.
    * Literally put all the issued money in your pocket
    */
    constructor() payable {
        totalSupply_ = INITIAL_SUPPLY;
        balances[msg.sender] = INITIAL_SUPPLY;
        emit Transfer(address(0), msg.sender, INITIAL_SUPPLY);
    }
}
