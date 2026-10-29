// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Real TOD-Balance: payFixed transfers a constant 50, payIfRich
// transfers a fixed 100 only when balance is high enough.  The two
// amounts compose non-commutatively across orders:
//
//   B in [150, 200) :
//     payFixed -> payIfRich : balance -= 50; (B-50 < 100, no second xfer)
//     payIfRich -> payFixed : balance -= 100; balance -= 50; -> -150
//   -> different final balance, TOD-Balance detected.
//
// Neither function modifies any *state variable*, so the pure
// public-state pass would report SUCCESSFUL.  The candidate finder
// has to recognise the virtual __balance write set, the harness has
// to emit the address(this).balance equality, and the EOA-deduct
// fallback in the transfer model has to fire so the asserts can
// observe the order-dependent decrement.
contract Bal {
    constructor() payable {}

    function payFixed(address payable to) public {
        if (to == address(this)) return; // skip self-transfer
        to.transfer(50);
    }

    function payIfRich(address payable to) public {
        if (to == address(this)) return; // skip self-transfer
        if (address(this).balance >= 150) {
            to.transfer(100);
        }
    }
}
