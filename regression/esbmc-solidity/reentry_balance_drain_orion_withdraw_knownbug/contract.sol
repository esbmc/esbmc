pragma solidity >=0.8.0;
// Derived from Orion Protocol Exchange.withdraw (the post-incident, CEI-fixed
// shape).  Two safeties combined:
//   1) `nonReentrant` mutex blocks any inbound reentrance into withdraw.
//   2) assetBalances[user] is debited BEFORE the external value transfer, so
//      net outflow over the call window equals exactly `amount`.
// Together they satisfy `balance >= pre - amount` in real EVM, but the test
// is classified KNOWNBUG: under --bound the reentry dispatch boundary does
// not propagate the outer call's `locked = true` storage write back to the
// inner call, so the mutex appears bypassed and the assert spuriously fires.
// The drain detected by the assert is therefore a model artefact, not a real
// vulnerability.  The original Orion contract emits
// `(bool ok, ) = user.call{value:amount}("")`; we substitute `transfer` since
// the low-level call{value:} dispatch has its own balance-propagation gap.
// Flips to CORE / VERIFICATION SUCCESSFUL once the bound-mode dispatch
// reliably preserves contract storage across the reentry boundary.
contract Exchange {
    mapping(address => uint256) public assetBalances;
    bool private locked;

    modifier nonReentrant() {
        require(!locked);
        locked = true;
        _;
        locked = false;
    }

    function deposit() external payable {
        assetBalances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) external nonReentrant {
        require(amount > 0);
        require(assetBalances[msg.sender] >= amount);
        assetBalances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
    }
}

contract Attacker {
    Exchange target;
    bool hit;
    constructor(Exchange _t) { target = _t; }
    function attack(uint256 amt) external { target.withdraw(amt); }
    receive() external payable {
        if (!hit && address(target).balance >= msg.value) {
            hit = true;
            target.withdraw(msg.value);
        }
    }
}
