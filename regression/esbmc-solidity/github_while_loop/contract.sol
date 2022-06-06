pragma solidity >=0.7.0 <0.9.0;
contract WhileContract {
  function while_loop() public {
    uint8 x = 2;
    while(x < 10) {
     ++x;
    }
    assert(x == 10);
  }
}
