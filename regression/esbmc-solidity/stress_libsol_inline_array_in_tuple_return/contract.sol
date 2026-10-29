contract C {
    uint[] a;
    function f() public returns (uint, uint) {
        a.push(1); a.push(2); a.push(3); a.push(0);
        return (a[3], [2,3,4][0]);
    }
}
