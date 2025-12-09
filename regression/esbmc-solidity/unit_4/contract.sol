// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TimeUnitTest {
    function testTimeUnits() public pure returns (bool) {
        // Solidity time literals
        uint oneSecond = 1 seconds;
        uint oneMinute = 1 minutes;
        uint oneHour = 1 hours;
        uint oneDay = 1 days;
        uint oneWeek = 1 weeks;

        // Assert each time unit to its value in seconds
        assert(oneSecond == 1);          // 1 second = 1 second
        assert(oneMinute == 60);         // 1 minute = 60 seconds
        assert(oneHour == 3600);         // 1 hour = 60 * 60 = 3600 seconds
        assert(oneDay == 86400);         // 1 day = 24 * 3600 = 86400 seconds
        assert(oneWeek == 604800);       // 1 week = 7 * 86400 = 604800 seconds

        return true;
    }
}