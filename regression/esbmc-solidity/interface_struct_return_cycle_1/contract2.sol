// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
import { A } from "./contract.sol";

interface IB {
    struct Order {
        uint256 x;
    }
    function bar(A a) external;
}
