// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Test {
    function createRandomNumber(uint256 max) public view returns (uint256) {
        require(max>0);
        return
            uint256(
                keccak256(
                    abi.encodePacked(
                        block.timestamp,
                        block.difficulty,
                        msg.sender
                    )
                )
            ) % max;
    }

    function checkRandomNumber() public view {
        uint x = createRandomNumber(128); // smaller than 128
        assert(x < 128);
    }
}
