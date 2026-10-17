// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.1;

interface DataFeed {
    function getData(address token) external returns (uint value);
}

// Tests try/catch with multiple catch clause types.
// ESBMC models try/catch as nondet: either success or any catch branch.
contract FeedConsumer {
    DataFeed feed;
    uint public errorCount;
    uint public lastValue;
    bool public lastSuccess;

    function rate(address token) public returns (uint value, bool success) {
        require(errorCount < 10);
        try feed.getData(token) returns (uint v) {
            lastValue = v;
            lastSuccess = true;
            return (v, true);
        } catch Error(string memory) {
            // revert with reason string
            errorCount++;
            lastSuccess = false;
            return (0, false);
        } catch (bytes memory) {
            // revert without reason / low-level error
            errorCount++;
            lastSuccess = false;
            return (0, false);
        }
    }

    function test(address token) public {
        (uint v, bool ok) = rate(token);
        // In all branches: ok == lastSuccess
        assert(ok == lastSuccess);
    }
}
