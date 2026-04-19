#include <iostream>
#include <vector>
#include <bitset>
#include <cassert>

// Basic logic gates as combination circuits
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

// Half Adder - adds two single bits
class HalfAdder
{
public:
  struct Result
  {
    bool sum;
    bool carry;
  };

  static Result add(bool a, bool b)
  {
    return {
      LogicGates::XOR(a, b), // Sum = A XOR B
      LogicGates::AND(a, b)  // Carry = A AND B
    };
  }
};

// Full Adder - adds two bits plus carry input
class FullAdder
{
public:
  struct Result
  {
    bool sum;
    bool carry_out;
  };

  static Result add(bool a, bool b, bool carry_in)
  {
    auto half1 = HalfAdder::add(a, b);
    auto half2 = HalfAdder::add(half1.sum, carry_in);

    return {
      half2.sum,                               // Final sum
      LogicGates::OR(half1.carry, half2.carry) // Carry out
    };
  }
};

// 4-bit Ripple Carry Adder
class RippleCarryAdder
{
public:
  struct Result
  {
    std::bitset<4> sum;
    bool carry_out;
  };

  static Result
  add(const std::bitset<4> &a, const std::bitset<4> &b, bool carry_in = false)
  {
    Result result;
    bool carry = carry_in;

    for (int i = 0; i < 4; i++)
    {
      auto fa_result = FullAdder::add(a[i], b[i], carry);
      result.sum[i] = fa_result.sum;
      carry = fa_result.carry_out;
    }

    result.carry_out = carry;
    return result;
  }
};

void testAdders()
{
  std::cout << "=== Adders Test ===\n";

  // Half Adder comprehensive test
  auto ha_00 = HalfAdder::add(false, false);
  assert(ha_00.sum == false && ha_00.carry == false); // 0+0 = 0, carry=0

  auto ha_01 = HalfAdder::add(false, true);
  assert(ha_01.sum == true && ha_01.carry == false); // 0+1 = 1, carry=0

  auto ha_10 = HalfAdder::add(true, false);
  assert(ha_10.sum == true && ha_10.carry == false); // 1+0 = 1, carry=0

  auto ha_11 = HalfAdder::add(true, true);
  assert(ha_11.sum == false && ha_11.carry == true); // 1+1 = 0, carry=1
  std::cout << "Half Adder (1+1): Sum=" << ha_11.sum
            << ", Carry=" << ha_11.carry << " ✓\n";

  // Full Adder comprehensive test
  auto fa_000 = FullAdder::add(false, false, false);
  assert(
    fa_000.sum == false && fa_000.carry_out == false); // 0+0+0 = 0, carry=0

  auto fa_001 = FullAdder::add(false, false, true);
  assert(fa_001.sum == true && fa_001.carry_out == false); // 0+0+1 = 1, carry=0

  auto fa_011 = FullAdder::add(false, true, true);
  assert(fa_011.sum == false && fa_011.carry_out == true); // 0+1+1 = 0, carry=1

  auto fa_111 = FullAdder::add(true, true, true);
  assert(fa_111.sum == true && fa_111.carry_out == true); // 1+1+1 = 1, carry=1
  std::cout << "Full Adder (1+1+1): Sum=" << fa_111.sum
            << ", Carry=" << fa_111.carry_out << " ✓\n";

  // 4-bit Adder test cases
  std::bitset<4> a1("0000"), b1("0000"); // 0 + 0 = 0
  auto rca1 = RippleCarryAdder::add(a1, b1);
  assert(rca1.sum == std::bitset<4>("0000") && rca1.carry_out == false);

  std::bitset<4> a2("0001"), b2("0001"); // 1 + 1 = 2
  auto rca2 = RippleCarryAdder::add(a2, b2);
  assert(rca2.sum == std::bitset<4>("0010") && rca2.carry_out == false);

  std::bitset<4> a3("1010"), b3("0110"); // 10 + 6 = 16 (0 with carry)
  auto rca3 = RippleCarryAdder::add(a3, b3);
  assert(rca3.sum == std::bitset<4>("0000") && rca3.carry_out == true);
  std::cout << "4-bit Adder (" << a3 << " + " << b3 << "): " << rca3.sum
            << ", Carry=" << rca3.carry_out << " ✓\n";

  std::bitset<4> a4("0111"), b4("0001"); // 7 + 1 = 8
  auto rca4 = RippleCarryAdder::add(a4, b4);
  assert(rca4.sum == std::bitset<4>("1000") && rca4.carry_out == false);

  std::bitset<4> a5("1111"), b5("0001"); // 15 + 1 = 16 (0 with carry)
  auto rca5 = RippleCarryAdder::add(a5, b5);
  assert(rca5.sum == std::bitset<4>("0000") && rca5.carry_out == true);

  std::cout << "All adder tests passed! ✓\n\n";
}

int main()
{
  try
  {
    testAdders();
  }
  catch (const std::exception &e)
  {
    std::cout << "TEST FAILED: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
