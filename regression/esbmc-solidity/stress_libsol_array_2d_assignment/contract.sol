contract C {
	function f(uint n) public pure returns (uint) {
		uint[][] memory a = new uint[][](2);
		for (uint i = 0; i < 2; ++i)
			a[i] = new uint[](3);
		a[1][1] = n;
		uint[] memory b = a[1];
		return b[1];
	}
}
// ----
// f(uint256): 42 -> 42
// KNOWNBUG: upstream k-induction disables the inductive step here ("cannot
// havoc what a loop writes through a pointer"), so only base cases run and
// the expected proof is never reached.
