// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

struct Vector2D {
    int x;
    int y;
}

library VectorLib {
    function magnitude(Vector2D memory v) internal pure {
        uint sum = uint(v.x * v.x + v.y * v.y); 
        uint len = sqrt(sum);                   

    }

    function sqrt(uint x) private pure returns (uint) {
        uint i = 0;
        while (i * i <= x) {
            i++;
        }
        return i - 1;
    }
}

contract VectorTest {
    function testMagnitude(int x, int y) public pure {
        Vector2D memory v = Vector2D(x, y);
        VectorLib.magnitude(v);
    }
}
