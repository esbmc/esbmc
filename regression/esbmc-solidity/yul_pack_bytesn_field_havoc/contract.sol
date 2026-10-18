// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// OVER-REACH GUARD: a `bytes4` field is a value type in storage but ESBMC
// models bytesN as a BytesStatic struct (not a bitvector), so the single-slot
// pack/unpack (which zero-extends a bitvector member) MUST NOT apply. The block
// must fall back to havoc. We assert the `[approx]` warning fires. If a future
// change routes bytesN through the raw bit-shift path, this test catches it.
contract H {
    struct P {
        bytes4 tag;
        uint224 x;
    }
    P s;

    function check(uint224 vx) public {
        assembly {
            sstore(s.slot, shl(32, vx))
        }
        uint224 rx;
        assembly {
            rx := shr(32, sload(s.slot))
        }
        assert(rx == vx);
    }
}
