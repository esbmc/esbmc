pragma solidity >=0.8.0;

/*
 * Minimal reproducer for: --k-induction --multi-property does not
 * terminate once all violable claims are discovered, and the final
 * verdict incorrectly prints VERIFICATION UNKNOWN despite "Bug found".
 *
 * Expected behaviour (post-fix): VERIFICATION FAILED at the base case,
 * driven by `balanceOf[target] += mintedAmount` overflow (same shape as
 * the classic mintToken CVE).
 */
contract MinTok {
    mapping(address => uint256) public balanceOf;
    uint256 public totalSupply;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function mintToken(address target, uint256 mintedAmount) public {
        require(msg.sender == owner);
        balanceOf[target] += mintedAmount;
        totalSupply += mintedAmount;
    }
}
