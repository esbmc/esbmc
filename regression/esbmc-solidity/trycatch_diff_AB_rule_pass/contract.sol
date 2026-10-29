pragma solidity >=0.8.0;
contract A { function test(uint256 x) external pure { require(x < 100, "A big"); } }
contract B { function test(uint256 x) external pure { require(x >= 100, "B small"); } }
contract Harness {
    A a; B b;
    function __ESBMC_reverted() internal returns (bool) {}
    constructor() { a = new A(); b = new B(); }
    function check(uint256 x) public {
        bool ra; bool rb;
        try a.test(x) { ra = false; } catch { ra = true; }
        require(!ra);                 // constrain to "A did not revert" (x<100)
        try b.test(x) { rb = false; } catch { rb = true; }
        assert(rb);                   // rule: A-ok => B reverts; holds on P for all x<100
    }
}
