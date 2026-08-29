pragma solidity >=0.5.26;

contract MyContract {
	function bool_literal() public {
		bool y = true;
		assert(y);
	}
}


// esbmc/esbmc#4715: the Solidity frontend synthesises the $call/$send/
// $transfer/$staticcall/$delegatecall members with unlocated returns, so the
// native return arm and the legacy round-trip disagreed on every contract.
// Comments go at the end: the committed .solast pins the line numbers above.
