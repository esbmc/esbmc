// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

// Differential verification (InvMut Diff(P,M,o,tb)) of a CEI reentrancy mutant.
// M1: C_mut.withdraw moves `balance[msg.sender] = 0` BELOW the external call
// (checks-effects-interactions violation -> real reentrancy bug).
//
// KNOWNBUG: a real EVM execution distinguishes P from M1 (reentrancy double-spend),
// so the boundary assertion SHOULD report VERIFICATION FAILED. ESBMC reports
// SUCCESSFUL instead: under --bound the low-level `.call` is a RETURN-0 stub that
// moves no ETH ($balance untouched) and does not re-enter, so without reentrancy
// P and M1's `withdraw` are observationally identical (same final state, only
// statement order differs). The mutant survives -> documented gap, not hidden.

contract C_ref {
    address public owner = msg.sender;
    uint256 public rewardRate = 100;
    mapping(address => uint256) public balance;
    mapping(address => uint256) public reward;
    mapping(address => uint256) public rateSnapshot;
    function deposit() external payable {
        balance[msg.sender] += msg.value;
        rateSnapshot[msg.sender] = rewardRate;
    }
    function setRewardRate(uint256 r) external { require(msg.sender == owner); rewardRate = r; }
    function claim() external returns (uint256 a) {
        a = balance[msg.sender] * rateSnapshot[msg.sender];
        reward[msg.sender] += a;
    }
    function withdraw() external {
        uint256 a = balance[msg.sender];
        require(a > 0);
        balance[msg.sender] = 0;                            //M1: move below the call
        (bool ok,) = msg.sender.call{value: a}(""); require(ok);
    }
}

contract C_mut {
    address public owner = msg.sender;
    uint256 public rewardRate = 100;
    mapping(address => uint256) public balance;
    mapping(address => uint256) public reward;
    mapping(address => uint256) public rateSnapshot;
    function deposit() external payable {
        balance[msg.sender] += msg.value;
        rateSnapshot[msg.sender] = rewardRate;
    }
    function setRewardRate(uint256 r) external { require(msg.sender == owner); rewardRate = r; }
    function claim() external returns (uint256 a) {
        a = balance[msg.sender] * rateSnapshot[msg.sender];
        reward[msg.sender] += a;
    }
    function withdraw() external {
        uint256 a = balance[msg.sender];
        require(a > 0);
        (bool ok,) = msg.sender.call{value: a}(""); require(ok);
        balance[msg.sender] = 0;                            //M1: moved below the call
    }
}

// differential verification
contract Harness {
    C_ref p= new C_ref(); C_mut m= new C_mut();
    constructor() {require(address(p).balance == address(m).balance); }
    function __ESBMC_reverted() internal returns (bool) {}
    function __ESBMC_assume(bool) internal pure {}

    function s0_deposit(uint256 __v, address k) public payable {
        p.balance(k);
        m.balance(k);
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
        p.withdraw(); bool pr = __ESBMC_reverted();
        m.withdraw(); bool mr = __ESBMC_reverted();
        // single differential assertion at the boundary tb = withdraw (state target),
        // revert-guarded: compare only on matched-revert paths.
        assert(pr != mr || address(p).balance == address(m).balance);
    }
}
