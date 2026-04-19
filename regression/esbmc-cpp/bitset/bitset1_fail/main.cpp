#include <iostream>
#include <vector>
#include <bitset>
#include <cassert>

class LogicGates
{
public:
  static bool AND(bool a, bool b)
  {
    return a && b;
  }
  static bool OR(bool a, bool b)
  {
    return a || b;
  }
  static bool NOT(bool a)
  {
    return !a;
  }
  static bool XOR(bool a, bool b)
  {
    return a != b;
  }
  static bool NAND(bool a, bool b)
  {
    return !(a && b);
  }
  static bool NOR(bool a, bool b)
  {
    return !(a || b);
  }
  static bool XNOR(bool a, bool b)
  {
    return a == b;
  }
};

void testBasicGates()
{
  // Test AND gate
  assert(LogicGates::AND(true, true) == true);
  assert(LogicGates::AND(true, false) == false);
  assert(LogicGates::AND(false, true) == false);
  assert(LogicGates::AND(false, false) == false);
  std::cout << "AND(1,1) = " << LogicGates::AND(true, true) << " ✓\n";

  // Test OR gate
  assert(LogicGates::OR(true, true) == true);
  assert(LogicGates::OR(true, false) == true);
  assert(LogicGates::OR(false, true) == true);
  assert(LogicGates::OR(false, false) == true); // should fail
  std::cout << "OR(0,1) = " << LogicGates::OR(false, true) << " ✓\n";

  // Test XOR gate
  assert(LogicGates::XOR(true, true) == false);
  assert(LogicGates::XOR(true, false) == true);
  assert(LogicGates::XOR(false, true) == true);
  assert(LogicGates::XOR(false, false) == false);
  std::cout << "XOR(1,1) = " << LogicGates::XOR(true, true) << " ✓\n";

  // Test NOT gate
  assert(LogicGates::NOT(true) == false);
  assert(LogicGates::NOT(false) == true);
  std::cout << "NOT(1) = " << LogicGates::NOT(true) << " ✓\n";

  // Test NAND gate
  assert(LogicGates::NAND(true, true) == false);
  assert(LogicGates::NAND(true, false) == true);
  assert(LogicGates::NAND(false, true) == true);
  assert(LogicGates::NAND(false, false) == true);

  // Test NOR gate
  assert(LogicGates::NOR(true, true) == false);
  assert(LogicGates::NOR(true, false) == false);
  assert(LogicGates::NOR(false, true) == false);
  assert(LogicGates::NOR(false, false) == true);

  // Test XNOR gate
  assert(LogicGates::XNOR(true, true) == true);
  assert(LogicGates::XNOR(true, false) == false);
  assert(LogicGates::XNOR(false, true) == false);
  assert(LogicGates::XNOR(false, false) == true);
}

int main()
{
  try
  {
    testBasicGates();
  }
  catch (const std::exception &e)
  {
    std::cout << "TEST FAILED: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
