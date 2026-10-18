// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
interface Vm {
  function fee(uint256) external; function chainId(uint256) external;
  function prevrandao(uint256) external; function txGasPrice(uint256) external;
  function coinbase(address) external;
}
abstract contract Test { Vm internal constant vm = Vm(0x7109709ECfa91a80626fF3989D68f67F5b1DD12D); }
contract S is Test {
  function test_fee() public { vm.fee(7); assert(block.basefee == 7); }
  function test_chainId() public { vm.chainId(99); assert(block.chainid == 99); }
  function test_prevrandao() public { vm.prevrandao(42); assert(block.prevrandao == 42); }
  function test_txGasPrice() public { vm.txGasPrice(3); assert(tx.gasprice == 3); }
  function test_coinbase() public { vm.coinbase(address(0xBEEF)); assert(block.coinbase == address(0xBEEF)); }
  function test_fee_wrong() public { vm.fee(7); assert(block.basefee == 8); }
}
