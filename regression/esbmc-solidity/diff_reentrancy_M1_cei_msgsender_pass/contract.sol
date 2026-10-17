// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

// Identical-pair companion of diff_reentrancy_M1_cei_msgsender_fail: both P and
// M are CEI-correct (zero before the external call), routed the natural way by
// `msg.sender == address(p)`/`address(m)`. The single boundary differential
// `assert(snap_p == snap_m)` holds (both expose 0 at the call instant) -> no
// false alarm. With the reentrant-msg.sender fix the callback fires and routes
// correctly, yet the assertion still passes -> VERIFICATION SUCCESSFUL.

contract C_ref {
    address public owner = msg.sender;
    uint256 public rewardRate = 100;
    mapping(address => uint256) public balance;
    mapping(address => uint256) public reward;
    mapping(address => uint256) public rateSnapshot;
    function deposit() external payable { balance[msg.sender] += msg.value; rateSnapshot[msg.sender] = rewardRate; }
    function setRewardRate(uint256 r) external { require(msg.sender == owner); rewardRate = r; }
    function claim() external returns (uint256 a) { a = balance[msg.sender] * rateSnapshot[msg.sender]; reward[msg.sender] += a; }
    function withdraw() external {
        uint256 a = balance[msg.sender];
        require(a > 0);
        balance[msg.sender] = 0;                            // CEI-correct
        (bool ok,) = msg.sender.call{value: a}(""); require(ok);
    }
}

contract C_mut {
    address public owner = msg.sender;
    uint256 public rewardRate = 100;
    mapping(address => uint256) public balance;
    mapping(address => uint256) public reward;
    mapping(address => uint256) public rateSnapshot;
    function deposit() external payable { balance[msg.sender] += msg.value; rateSnapshot[msg.sender] = rewardRate; }
    function setRewardRate(uint256 r) external { require(msg.sender == owner); rewardRate = r; }
    function claim() external returns (uint256 a) { a = balance[msg.sender] * rateSnapshot[msg.sender]; reward[msg.sender] += a; }
    function withdraw() external {
        uint256 a = balance[msg.sender];
        require(a > 0);
        balance[msg.sender] = 0;                                  // identical
        (bool ok,) = msg.sender.call{value: a}(""); require(ok);
    }
}

// differential verification — one sync-wrapper per public/external function;
// the single differential assertion lives only in the boundary wrapper s3_withdraw.
// Callback routed by the NATURAL `msg.sender == address(p)` (no workaround flag).
contract Harness {
    C_ref p = new C_ref(); C_mut m = new C_mut();
    uint256 snap_p; uint256 snap_m;
    constructor() { require(address(p).balance == address(m).balance); }
    function __ESBMC_reverted() internal returns (bool) {}
    function __ESBMC_assume(bool) internal pure {}

    receive() external payable {
        if (msg.sender == address(p))      snap_p = p.balance(address(this));
        else if (msg.sender == address(m)) snap_m = m.balance(address(this));
    }

    function s0_deposit(uint256 __v) public payable {
        p.deposit{value: __v}();
        m.deposit{value: __v}();
    }
    function s1_setRewardRate(uint256 a0) public {
        p.setRewardRate(a0);
        m.setRewardRate(a0);
    }
    function s2_claim() public {
        p.claim();
        m.claim();
    }
    function s3_withdraw() public {
        p.withdraw();
        m.withdraw();
        assert(snap_p == snap_m);   // single differential assertion at boundary tb = withdraw
    }
}
