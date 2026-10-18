// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// tx.origin and msg.sender must be modelled as INDEPENDENT symbolic values: a
// contract calling `setACL` can arrange tx.origin == ACL (the authority) while
// msg.sender is the intermediate contract, so tx.origin-based authentication
// (SWC-115) is bypassable. The oracle assert `msg.sender == ACL` must therefore
// be violable even though `require(tx.origin == ACL)` holds. Regression pin:
// guards against a future change that conflates tx.origin with msg.sender,
// which would silently mask this access-control bug (VERIFICATION SUCCESSFUL).
contract A {
    address ACL;
    constructor() { ACL = msg.sender; }
    function setACL(address x) external {
        require(tx.origin == ACL);   // tx.origin auth — the injected bug
        assert(msg.sender == ACL);   // oracle: immediate caller must be authority
        ACL = x;
    }
}
