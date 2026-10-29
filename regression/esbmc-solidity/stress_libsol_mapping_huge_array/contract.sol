contract C {
    mapping(uint => uint[2**100]) x;
    function f() public view returns (uint) {
        return 1;
    }
}
