// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: WinToken
// Pair:     approve vs increaseApproval
// Mode:     race

// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// TOD classification helpers.  An assertion failure inside one
// of these functions tells the user which TOD category fired.
function __tod_race_check(bool cond) pure {
    assert(cond); // TOD-Race: non-commutative state update
}
function __tod_balance_check(bool cond) pure {
    assert(cond); // TOD-Balance: order-dependent ETH movement
}

// ESBMC intrinsic stubs (the frontend ignores the bodies).
function __ESOL_nondet_state_forward(WinToken c) {
    // replaced by ESBMC with a bounded nondet-dispatch loop
    // over c's public/external methods.
}
function __ESOL_deep_copy(WinToken src) pure returns (WinToken) {
    // replaced by ESBMC with _ESBMC_clone_WinToken: per-field deep copy of *src
    // into a fresh instance with a distinct $address and
    // independent heap-allocated array buffers.
    return src;
}

// ===== Target contract =====
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

// ===== TOD harness =====
// ----- approve vs increaseApproval -----
// Shared state variables (touched by both):
//   - allowed
contract TOD_approve_increaseApproval {
    function test(
        address a__spender, uint256 a__value,
        address b__spender, uint256 b__addedValue
    ) public {
        WinToken c1 = new WinToken();
        __ESOL_nondet_state_forward(c1);
        WinToken c2 = __ESOL_deep_copy(c1);
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs approve then increaseApproval
        c1.approve(a__spender, a__value);
        c1.increaseApproval(b__spender, b__addedValue);

        // Order 2: c2 runs increaseApproval then approve
        c2.increaseApproval(b__spender, b__addedValue);
        c2.approve(a__spender, a__value);

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.allowed(a__spender, a__spender) == c2.allowed(a__spender, a__spender));
        __tod_race_check(c1.allowed(a__spender, b__spender) == c2.allowed(a__spender, b__spender));
        __tod_race_check(c1.allowed(a__spender, address(this)) == c2.allowed(a__spender, address(this)));
        __tod_race_check(c1.allowed(b__spender, a__spender) == c2.allowed(b__spender, a__spender));
        __tod_race_check(c1.allowed(b__spender, b__spender) == c2.allowed(b__spender, b__spender));
        __tod_race_check(c1.allowed(b__spender, address(this)) == c2.allowed(b__spender, address(this)));
        __tod_race_check(c1.allowed(address(this), a__spender) == c2.allowed(address(this), a__spender));
        __tod_race_check(c1.allowed(address(this), b__spender) == c2.allowed(address(this), b__spender));
        __tod_race_check(c1.allowed(address(this), address(this)) == c2.allowed(address(this), address(this)));
    }
}

