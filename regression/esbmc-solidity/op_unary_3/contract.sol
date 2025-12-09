pragma solidity ^0.8.0;

contract RationalTest {
    function test() public pure {
        // Rational constants
        uint a = 0.1 ether; // Should be 0.1 * 1e18 = 1e17
        uint b = 10 * 0.5 ether; // 10 * 0.5 * 1e18 = 5e18
        uint c = uint(0.2 * 1 days); // 0.2 * 86400 = 17280 seconds

        // Simple checks
        assert(a == 1e17);         // 0.1 ether is 1e17 wei
        assert(b == 5e18);         // 5 ether
        assert(c == 17280);        // 17280 seconds

        // Purposely fail with rational math
        uint d = 5 * 0.2 ether;    // Should be 1e18
        assert(d == 1 ether);      // Check if 5 * 0.2 = 1.0 correctly

    }
}
