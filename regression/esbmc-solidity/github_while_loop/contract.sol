pragma solidity >=0.7.0 <0.9.0;
contract WhileContract {
  function while_underflow() public {
    uint8 x = 2;
    while(x < 3) {
     --x;
    }
    assert(x == 0);
  }
}
