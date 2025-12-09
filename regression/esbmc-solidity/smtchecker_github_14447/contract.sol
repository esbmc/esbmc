// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
contract C
{
	uint x;
	uint y;

	function condition() private returns(bool) {
		x = (x + 1) % 2;
		return (x == 1);
	}

	function f() public {
		require(x == 0);
		require(y == 0);
		for (; condition();) {
			++y;
		}
		assert(y == 1);
	}
}
