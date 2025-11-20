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

// 2-to-1 Multiplexer
class Multiplexer2to1
{
public:
  static bool select(bool input0, bool input1, bool select)
  {
    // Output = (NOT(select) AND input0) OR (select AND input1)
    return LogicGates::OR(
      LogicGates::AND(LogicGates::NOT(select), input0),
      LogicGates::AND(select, input1));
  }
};

// 4-to-1 Multiplexer
class Multiplexer4to1
{
public:
  static bool
  select(bool in0, bool in1, bool in2, bool in3, bool sel0, bool sel1)
  {
    // Use two 2-to-1 muxes to build a 4-to-1 mux
    bool mux0_out = Multiplexer2to1::select(in0, in1, sel0);
    bool mux1_out = Multiplexer2to1::select(in2, in3, sel0);
    return Multiplexer2to1::select(mux0_out, mux1_out, sel1);
  }
};

void testMultiplexer()
{
  std::cout << "=== Multiplexer Test ===\n";

  // 2-to-1 Mux comprehensive test
  // When select = 0, output should be input0
  assert(Multiplexer2to1::select(false, true, false) == false);
  assert(Multiplexer2to1::select(true, false, false) == true);

  // When select = 1, output should be input1
  assert(Multiplexer2to1::select(false, true, true) == true);
  assert(Multiplexer2to1::select(true, false, true) == false);
  std::cout << "2-to-1 Mux (inputs: 0,1; select: 1): "
            << Multiplexer2to1::select(false, true, true) << " ✓\n";

  // 4-to-1 Mux comprehensive test
  // Test all combinations of select bits (sel1, sel0)
  // sel1=0, sel0=0 -> should select input0
  assert(
    Multiplexer4to1::select(true, false, false, false, false, false) == true);
  assert(
    Multiplexer4to1::select(false, true, true, true, false, false) == false);

  // sel1=0, sel0=1 -> should select input1
  assert(
    Multiplexer4to1::select(false, true, false, false, true, false) == true);
  assert(
    Multiplexer4to1::select(true, false, true, true, true, false) == false);

  // sel1=1, sel0=0 -> should select input2
  assert(
    Multiplexer4to1::select(false, false, true, false, false, true) == true);
  assert(
    Multiplexer4to1::select(true, true, false, true, false, true) == false);

  // sel1=1, sel0=1 -> should select input3
  assert(
    Multiplexer4to1::select(false, false, false, true, true, true) == true);
  assert(Multiplexer4to1::select(true, true, true, false, true, true) == false);

  bool result4 = Multiplexer4to1::select(false, true, false, true, true, false);
  std::cout << "4-to-1 Mux (inputs: 0,1,0,1; select: 01): " << result4
            << " ✓\n";

  std::cout << "All multiplexer tests passed! ✓\n\n";
}

int main()
{
  try
  {
    testMultiplexer();
  }
  catch (const std::exception &e)
  {
    std::cout << "TEST FAILED: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
