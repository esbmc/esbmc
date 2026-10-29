pragma abicoder               v2;


contract C {
    function f(string[] calldata a)
        external
        returns (uint256, uint256, uint256, string memory)
    {
        string memory s1 = a[0];
        bytes memory m1 = bytes(s1);
        return (a.length, m1.length, uint8(m1[0]), s1);
    }
}
// ----
// f(string[]): 0x20, 0x1, 0x20, 0x2, hex"6162000000000000000000000000000000000000000000000000000000000000" -> 1, 2, 97, 0x80, 2, "ab"
// `a[0]` may be the empty string, so `m1[0]` is out of bounds:
// --incremental-bmc reaches it at k = 5. The SUCCESSFUL this test once
// expected came from an inductive step that closed at k = 4 before the base
// case got there.
