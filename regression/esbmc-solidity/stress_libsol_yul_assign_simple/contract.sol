contract C {
    function f() public pure returns (bool) {
        uint a = 42;
        uint b;
        assembly { b := a }
        assert(b == 42);
        return true;
    }
}
