// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for monomorphized library callee whose storage parameter
// AST id was truncated to int16_t when registered in the symbol table.
//
// The monomorphizer clones the callee per (callee, fn-ptr-arg) tuple and
// renames its param ids via fresh_id_offset() (base 1e8 + 1e5 per clone).
// get_local_var_decl_name registered those ids using
// `ast_node["id"].get<std::int16_t>()`, which truncated 1e8+ to a small
// wraparound number. The storage-ref bridge in get_function_definition
// then looked the param up using `get<int>()` and missed it, aborting
// with "storage-ref bridge: param symbol ... not found" before symex
// ever ran.
//
// Reproduced on 1inch/farming's FarmingLib._farmedPerToken path
// (library using-for call to UserAccounting.farmedPerToken(Info storage,
// ..., function internal, function internal)). The fix widens id
// registration from int16_t to int so remapped clone ids survive.

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
        // Any outcome is fine; the point is that the frontend no longer
        // aborts during Converting on the monomorphized `ping__mono_*`
        // storage-ref bridge lookup.
        Outer.call(d);
        assert(true);
    }
}
