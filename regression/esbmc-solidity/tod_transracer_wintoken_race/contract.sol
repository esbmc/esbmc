// SPDX-License-Identifier: MIT
// TransRacer Figure 1(a) WinToken motivational example — "whoever grew
// the pot past the threshold first wins":
//   - addPot(v) grows the shared pot by `v`.
//   - claimWin(t) writes msg.sender to `winner` iff pot >= t.
// Running addPot before claimWin vs claimWin before addPot can leave
// `winner` in different states (a front-runner's deposit decides who
// claims the win), so the pair exhibits a TOD-Race over state
// variable `winner`.
pragma solidity >=0.8.0;

contract WinToken {
    uint256 public pot;
    address public winner;

    function addPot(uint256 amount) public {
        pot = pot + amount;
    }

    function claimWin(uint256 threshold) public {
        if (pot >= threshold) {
            winner = msg.sender;
        }
    }
}
