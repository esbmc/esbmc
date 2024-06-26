pragma solidity >=0.5.0;

contract Test {
    function testCalculateGas() public {
        uint256 gasStart = gasleft();
        uint256 gasEnd = gasleft();
        require(gasStart > gasEnd);
        require(tx.gasprice > 0);
        uint256 gasUsed = (gasStart - gasEnd) * tx.gasprice;
        assert(gasUsed != 0);
    }
}
