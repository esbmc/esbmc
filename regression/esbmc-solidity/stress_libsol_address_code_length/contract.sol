contract C {
    function f() public view returns (uint) { return address(this).code.length; }
    function g() public view returns (uint) { return address(0).code.length; }
    function h(address a) public view returns (uint) { return a.code.length; }
}
