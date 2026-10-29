pragma abicoder               v2;


contract C {
    function f1(bytes[1] calldata a)
        external
        returns (uint256, uint256, uint256, uint256)
    {
        bytes memory a0 = a[0];
        require(a0.length > 2);
        return (a0.length, uint8(a0[0]), uint8(a0[1]), uint8(a0[2]));
    }

    function f2(bytes[1] calldata a, bytes[1] calldata b)
        external
        returns (uint256, uint256, uint256, uint256, uint256, uint256, uint256)
    {
        bytes memory a0 = a[0];
        bytes memory b0 = b[0];
        require(a0.length > 2 && b0.length > 1);
        return (
            a0.length,
            uint8(a0[0]),
            uint8(a0[1]),
            uint8(a0[2]),
            b0.length,
            uint8(b0[0]),
            uint8(b0[1])
        );
    }

    function g1(bytes[2] calldata a)
        external
        returns (
            uint256,
            uint256,
            uint256,
            uint256,
            uint256,
            uint256,
            uint256,
            uint256
        )
    {
        bytes memory a0 = a[0];
        bytes memory a1 = a[1];
        require(a0.length > 2 && a1.length > 2);
        return (
            a0.length,
            uint8(a0[0]),
            uint8(a0[1]),
            uint8(a0[2]),
            a1.length,
            uint8(a1[0]),
            uint8(a1[1]),
            uint8(a1[2])
        );
    }

    function g2(bytes[] calldata a) external returns (uint256[8] memory) {
        require(a.length > 1);
        bytes memory a0 = a[0];
        bytes memory a1 = a[1];
        require(a0.length > 1 && a1.length > 2);
        return [
            a.length,
            a0.length,
            uint8(a0[0]),
            uint8(a0[1]),
            a1.length,
            uint8(a1[0]),
            uint8(a1[1]),
            uint8(a1[2])
        ];
    }
}

// via yul disabled because of stack issues.

// ====
// compileViaYul: false
// ----
// f1(bytes[1]): 0x20, 0x20, 0x3, hex"0102030000000000000000000000000000000000000000000000000000000000" -> 0x3, 0x1, 0x2, 0x3
// f2(bytes[1],bytes[1]): 0x40, 0xa0, 0x20, 0x3, hex"0102030000000000000000000000000000000000000000000000000000000000", 0x20, 0x2, hex"0102000000000000000000000000000000000000000000000000000000000000" -> 0x3, 0x1, 0x2, 0x3, 0x2, 0x1, 0x2
// g1(bytes[2]): 0x20, 0x40, 0x80, 0x3, hex"0102030000000000000000000000000000000000000000000000000000000000", 0x3, hex"0405060000000000000000000000000000000000000000000000000000000000" -> 0x3, 0x1, 0x2, 0x3, 0x3, 0x4, 0x5, 0x6
// g1(bytes[2]): 0x20, 0x40, 0x40, 0x3, hex"0102030000000000000000000000000000000000000000000000000000000000" -> 0x3, 0x1, 0x2, 0x3, 0x3, 0x1, 0x2, 0x3
// g2(bytes[]): 0x20, 0x2, 0x40, 0x80, 0x2, hex"0102000000000000000000000000000000000000000000000000000000000000", 0x3, hex"0405060000000000000000000000000000000000000000000000000000000000" -> 0x2, 0x2, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6
