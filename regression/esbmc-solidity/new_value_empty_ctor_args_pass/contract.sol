// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression for `new C{value: x}()` (empty constructor args).  Before the
// fix in solidity_convert_expr.cpp::FunctionCallOptions, the conversion
// path took expr["expression"]["argumentTypes"][0] without bounds-checking
// the array — empty argument list triggered a JSON out-of-range and
// core-dumped the frontend.
contract Bal {
    constructor() payable {}
    function balanceOf() public view returns (uint) {
        return address(this).balance;
    }
}

contract Caller {
    function deploy() public {
        Bal b = new Bal{value: 100}();
        // After the auto-bind fix, b.balanceOf() executes the callee body
        // even in unbound mode; combined with the {value:} crediting we
        // expect the new instance to start with 100.
        assert(b.balanceOf() == 100);
    }
}
