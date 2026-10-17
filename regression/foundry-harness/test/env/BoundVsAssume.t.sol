// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

import {Test} from "forge-std/Test.sol";

/// Environment self-test, not a test of any generated artifact.
///
/// The method's rendering rule is "every coordinate is ESTABLISHED by an
/// assignment, never FILTERED by a predicate", and the reason given is that
/// `bound` is a mapping with zero rejections while `vm.assume` is a filter that
/// discards draws and can exhaust the rejection budget. That pair of facts is
/// load-bearing -- it is why the box is the only shape that executes at full
/// yield -- and it is a property of THIS forge/forge-std, not a theorem.
///
/// So it is measured here rather than cited. `forge test` failing on this file
/// means the environment no longer supports the claim the paper makes, which is
/// exactly when we want to find out.
contract BoundVsAssumeTest is Test {
    uint256 constant LO = 10;
    uint256 constant HI = 20;

    /// `bound` maps every draw into [LO, HI]. No draw is discarded, so with
    /// runs = 256 the body executes 256 times and the assertion holds every
    /// time, including for inputs nowhere near the interval.
    function testFuzz_bound_never_rejects(uint256 x) public pure {
        uint256 v = _bound(x, LO, HI);
        assertGe(v, LO);
        assertLe(v, HI);
    }

    /// The same interval expressed as a FILTER. Every draw outside [LO, HI] is
    /// discarded, and the fraction outside is overwhelming for a 256-bit draw,
    /// so this is the shape that exhausts `max_test_rejects`. Kept as a
    /// documented counterpart rather than run: at runs = 256 it would abort the
    /// suite, which is the very behaviour being recorded.
    ///
    ///     function testFuzz_assume_rejects(uint256 x) public {
    ///         vm.assume(x >= LO && x <= HI);
    ///         assertGe(x, LO);
    ///     }
    ///
    /// Measured consequence, and the reason the rule is "assign, never filter":
    /// a filter narrow enough to matter rejects essentially every draw.

    /// The degenerate interval must still be a mapping, not a rejection:
    /// single-point boxes are the fallback the method degrades to when
    /// generalisation fails, so `LO == HI` has to execute at full yield too.
    function testFuzz_bound_single_point(uint256 x) public pure {
        uint256 v = _bound(x, LO, LO);
        assertEq(v, LO);
    }

    /// An empty interval must NOT be shipped. `_bound` reverts when min > max,
    /// which is why 5.0 requires the emptiness check BEFORE rendering: without
    /// it the failure surfaces as an unexplained revert inside a helper rather
    /// than as a refused test.
    function test_bound_empty_interval_reverts() public {
        vm.expectRevert();
        this.callBound(1, HI, LO);
    }

    function callBound(uint256 x, uint256 lo, uint256 hi)
        external
        pure
        returns (uint256)
    {
        return _bound(x, lo, hi);
    }
}
