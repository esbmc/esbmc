contract C {
    uint constant L = 3;
    uint constant L2 = ((2) + 1);
    uint[(L) + L2] a;
    function f() public view returns (uint) {
        return a.length;
    }
}
