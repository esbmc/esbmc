// SPDX-License-Identifier: MIT
// TransRacer Figure 1(a) WinToken — faithful upgrade of the paper's
// code snippet (pragma ^0.4.x) to ^0.8.0.
//
// Paper's original race pair of interest is `approve(spender, v)` vs
// `transferFrom(from, to, v)` (SWC-114 "ERC20 approve race": a malicious
// spender front-runs the owner's intent to change its allowance, draining
// both the old and the new value).  That bug requires pre-existing state
// on mainnet (non-zero `balances[from]` and `allowed[from][spender]`);
// starting from the zero Initial State, both orderings' transferFrom
// revert on `require(_value <= balances[_from])`, so the race is not
// manifest under IS.  This corresponds exactly to the paper's IS/US
// distinction — TransRacer relies on an Updated-State snapshot for
// approve/transferFrom.
//
// For a race pair that IS manifest under the fresh Initial State on this
// same contract, use `approve` vs `increaseApproval` — both write the
// exact same `allowed[msg.sender][_spender]` slot but in a non-commutative
// way (approve overwrites, increaseApproval reads-then-adds).  This is the
// pair the regression verifies.
//
// Differences from the paper snippet:
//   * SafeMath `.add` / `.sub` dropped — 0.8 has built-in overflow checks.
//   * `onlyOwner` / `lockTransfer` / `owner` made explicit (the snippet
//     referenced them without showing the decls).
//   * `balances` / `allowed` promoted to `public` so the TOD harness
//     can read them via generated getters.
//   * `transferFrom` body fleshed out (the snippet ends with `...`).
//   * Explicit `public` visibility on all functions (required since 0.5).
pragma solidity >=0.8.0;

contract WinToken {
    address public owner;
    bool public lockTransfer;
    uint256 public totalSupply;
    mapping(address => uint256) public balances;
    mapping(address => mapping(address => uint256)) public allowed;

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function setTransferLock(bool _set) public onlyOwner {
        lockTransfer = _set;
    }

    function mint(address _to, uint256 _amount) public onlyOwner {
        totalSupply = totalSupply + _amount;
        balances[_to] = balances[_to] + _amount;
    }

    function approve(address _spender, uint256 _value) public {
        allowed[msg.sender][_spender] = _value;
    }

    function increaseApproval(address _spender, uint256 _addedValue) public {
        allowed[msg.sender][_spender] =
            allowed[msg.sender][_spender] + _addedValue;
    }

    function transferFrom(address _from, address _to, uint256 _value) public {
        require(lockTransfer == false, "locked");
        require(_to != address(0), "zero to");
        require(_value <= balances[_from], "balance");
        require(_value <= allowed[_from][msg.sender], "allowance");
        allowed[_from][msg.sender] = allowed[_from][msg.sender] - _value;
        balances[_from] = balances[_from] - _value;
        balances[_to] = balances[_to] + _value;
    }
}
