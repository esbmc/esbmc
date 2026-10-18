// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual to pass: same monomorphized-storage-param pattern, with an
// unconditional assert(false) placed ahead of the library call so that
// if the frontend no longer aborts during Converting we actually reach
// symex and produce VERIFICATION FAILED rather than slicing to zero.

library Inner {
    struct Info {
        uint256 x;
    }
    function ping(
        Info storage info,
        function() internal view returns (uint256) cb
    ) internal view returns (uint256) {
        return info.x + cb();
    }
}

library Outer {
    using Inner for Inner.Info;

    struct Data {
        Inner.Info inner;
    }

    function _one() private pure returns (uint256) {
        return 1;
    }

    function call(Data storage d) internal view returns (uint256) {
        return d.inner.ping(_one);
    }
}

contract C {
    Outer.Data d;

    function go() external view {
        assert(false);
        // Still lowered by the frontend so the monomorphized aux
        // callee `ping__mono_*` is instantiated and its storage-ref
        // bridge lookup runs at Converting time.
        Outer.call(d);
    }
}
