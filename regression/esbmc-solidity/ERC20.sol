// SPDX-License-Identifier: MIT
// Simplified ERC20 model for ESBMC verification
// Based on OpenZeppelin Contracts v4.6 (token/ERC20/ERC20.sol)
//
// Simplifications for ESBMC compatibility:
//   - No interface/abstract contract inheritance (IERC20, IERC20Metadata, Context)
//   - No unchecked{} blocks (ESBMC checks overflow at verification level)
//   - No type(uint256).max (unsupported syntax)
//   - No _msgSender() — uses msg.sender directly
//   - No _beforeTokenTransfer/_afterTokenTransfer hooks (empty in base)
//   - No increaseAllowance/decreaseAllowance (convenience functions)
//   - No _spendAllowance (inlined into transferFrom)
//
// Usage: copy this file into your test directory and include it in a single
// Solidity file with your contract (ESBMC currently requires single-file input).
// Then generate the combined .solast with: solc --ast-compact-json contract.sol
pragma solidity >=0.8.0;

contract ERC20 {
    mapping(address => uint256) internal _balances;
    mapping(address => mapping(address => uint256)) internal _allowances;
    uint256 internal _totalSupply;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor() {}

    function totalSupply() public view virtual returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view virtual returns (uint256) {
        return _balances[account];
    }

    function transfer(address to, uint256 amount) public virtual returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }

    function allowance(address owner, address spender) public view virtual returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public virtual returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public virtual returns (bool) {
        uint256 currentAllowance = _allowances[from][msg.sender];
        require(currentAllowance >= amount);
        _approve(from, msg.sender, currentAllowance - amount);
        _transfer(from, to, amount);
        return true;
    }

    function _transfer(address from, address to, uint256 amount) internal virtual {
        require(from != address(0));
        require(to != address(0));
        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount);
        _balances[from] = fromBalance - amount;
        _balances[to] += amount;
        emit Transfer(from, to, amount);
    }

    function _mint(address account, uint256 amount) internal virtual {
        require(account != address(0));
        _totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }

    function _burn(address account, uint256 amount) internal virtual {
        require(account != address(0));
        uint256 accountBalance = _balances[account];
        require(accountBalance >= amount);
        _balances[account] = accountBalance - amount;
        _totalSupply -= amount;
        emit Transfer(account, address(0), amount);
    }

    function _approve(address owner, address spender, uint256 amount) internal virtual {
        require(owner != address(0));
        require(spender != address(0));
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
}
