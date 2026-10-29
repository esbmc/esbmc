// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

// InvMut behavioral differential of a CEI reentrancy mutant M1.
// M1: C_mut.withdraw moves `balance[msg.sender] = 0` BELOW the external
// `.call`, so during the reentrant callback the un-zeroed balance is still
// visible (real reentrancy double-spend window). P (C_ref) is CEI-correct.
//
// Detection is purely behavioral: the harness IS the call recipient, and
// its receive() snapshots `balance[msg.sender]` AT the call instant for
// each contract (routed by a context flag, since the reentrant msg.sender
// is the dispatch-singleton address, not address(p)). The single boundary
// assertion `assert(snap_p == snap_m)` then fails because P exposes 0 and
// M1 exposes v.
//
// Requires the bounded-nondet contract $balance init (_ESBMC_nondet_init_
// balance, [0,2^128)): a full-range nondet would let the deposit overflow
// so the `.call` funding check spuriously fails and the callback never
// fires (the bug this test guards against). Expected: VERIFICATION FAILED.
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
        uint256 a = balance[msg.sender]; require(a > 0);
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
        uint256 a = balance[msg.sender]; require(a > 0);
        (bool ok,) = msg.sender.call{value: a}(""); require(ok);   // M1: call BEFORE zero
        balance[msg.sender] = 0;
    }
}
contract Harness {
    C_ref p = new C_ref(); C_mut m = new C_mut();
    uint256 snap_p; uint256 snap_m; uint8 which;
    function __ESBMC_assume(bool) internal pure {}
    receive() external payable {
        if (which == 1) snap_p = p.balance(address(this));
        else if (which == 2) snap_m = m.balance(address(this));
    }
    function run() public payable {
        p.deposit{value: 5}(); m.deposit{value: 5}();
        which = 1; p.withdraw();
        which = 2; m.withdraw();
        assert(snap_p == snap_m);            // behavioral differential at withdraw boundary
    }
}
