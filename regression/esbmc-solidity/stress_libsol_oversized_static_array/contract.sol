
contract C {
    struct S {
        uint[2 ** 253] a;
    }
    S d;
    function f() public view returns (uint) {
        S storage x = d;
        return x.a[0];
    }
}
