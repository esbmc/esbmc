contract C {
    uint constant L = 5;
    uint constant LEN = L + 4 * L;
    uint[LEN] ids;
    function f() public view returns (uint) {
        return ids.length;
    }
}
