// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Item 2 — the 2c no-inflation guard (adversarial, strongest form).
// Universe = 4 edges (setX g/!g, setY g/!g), all reachable under
// --contract C => no-fixture run is Branches : 4 / Reached : 4 / 100%.
// covered.json pre-covers ALL 4 edges, so EVERY assert is skipped and
// the instrumented set is empty. A denominator derived from the
// instrumented set would then be 0 => "No branch detected" (a silent
// total collapse). Item 2c keeps the denominator at the full static
// universe (4) and credits all 4 from the covered-set =>
// Branches : 4 / Reached : 4 / 100% — identical to the no-fixture run,
// at zero SMT cost. The merge is a no-op (no new edge witnessed), so
// covered.json is byte-stable across runs (safe committed fixture).
// The KNOWNBUG dual cov_jsonset_inflation_knownbug pins the regressed
// `No branch detected` so a 2c regression is caught by a KNOWNBUG flip.
contract C {
    uint256 public x;
    uint256 public y;
    function setX(uint256 v) public {
        if (v > 10) {
            x = v;
        }
    }
    function setY(uint256 w) public {
        if (w > 20) {
            y = w;
        }
    }
}
