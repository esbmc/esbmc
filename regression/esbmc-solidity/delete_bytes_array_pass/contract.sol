// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Bug D regression-lock (was bytes[] delete crash, root cause: gen_zero
// produced ARRAY_OF(nil) for the BytesDynamic-element infinite array).
// Pre-fix: symex core dumps on the assignment.  Post-fix: delete only
// resets the length companion (data array left as-is, since reads past
// length are OOB by Solidity semantics).
//
// Note: `arr.push()` (no args) on `bytes[]` triggers an unrelated
// pre-existing symex crash (assigns `nil` into the data slot in
// `arr.push` lowering — independent of delete). To isolate the delete
// behaviour, this test uses `require(arr.length == N)` to pin the
// entry-state length without going through push.
contract C {
    bytes[] arr;

    function f() public {
        require(arr.length == 5);
        assert(arr.length == 5);
        delete arr;
        assert(arr.length == 0);
    }
}
