// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

// Identical-pair counterpart of diff_reentrancy_M1_cei_observer_fail:
// here C_mut is ALSO CEI-correct (zeroes balance BEFORE the external call),
// so P and M are behaviorally identical. The same receive()-observer
// behavioral differential `assert(snap_p == snap_m)` must hold (both expose
// 0 at the call instant) -> no false alarm. Expected: VERIFICATION
// SUCCESSFUL. Together with the _fail dual this pins both directions of the
// reentrancy differential under the bounded-nondet $balance fix.
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
        balance[msg.sender] = 0;                                   // identical: CEI-correct
        (bool ok,) = msg.sender.call{value: a}(""); require(ok);
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
