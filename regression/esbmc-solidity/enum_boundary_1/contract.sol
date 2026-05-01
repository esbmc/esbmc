// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract EnumTest {
    enum Status { Pending, Active, Completed, Cancelled }

    function test_enum() public pure {
        Status s = Status.Pending;
        assert(uint(s) == 0);

        s = Status.Active;
        assert(uint(s) == 1);

        s = Status.Completed;
        assert(uint(s) == 2);

        s = Status.Cancelled;
        assert(uint(s) == 3);

        // Comparison
        assert(Status.Pending != Status.Active);
        assert(Status.Active == Status.Active);

        // Assignment and re-check
        Status a = Status.Completed;
        Status b = Status.Completed;
        assert(a == b);
    }

    function test_enum_in_struct() public pure {
        Status s = Status.Active;
        uint8 val = uint8(s);
        assert(val == 1);

        // Round-trip: enum -> uint -> compare
        Status t = Status.Cancelled;
        assert(uint(t) > uint(Status.Completed));
    }
}
