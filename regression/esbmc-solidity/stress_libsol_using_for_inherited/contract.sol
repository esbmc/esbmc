library Lib {
    function foo(uint256 v) internal pure returns (uint256) { return v + 42; }
}
contract A { using Lib for uint256; }
contract B is A {
    using Lib for uint256;
    function bar(uint256 v) public pure returns (uint256) { return v.foo(); }
}
