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

// 2-to-4 Decoder
class Decoder2to4
{
public:
  struct Output
  {
    bool out0, out1, out2, out3;
  };

  static Output decode(bool a, bool b, bool enable = true)
  {
    if (!enable)
    {
      return {false, false, false, false};
    }

    bool not_a = LogicGates::NOT(a);
    bool not_b = LogicGates::NOT(b);

    return {
      LogicGates::AND(not_a, not_b), // 00
      LogicGates::AND(not_a, b),     // 01
      LogicGates::AND(a, not_b),     // 10
      LogicGates::AND(a, b)          // 11
    };
  }
};

void testDecoder()
{
  std::cout << "=== Decoder Test ===\n";

  // Test 2-to-4 decoder with all input combinations
  // Input 00 -> output should be 0001 (only out0 = 1)
  auto dec_00 = Decoder2to4::decode(false, false);
  assert(
    dec_00.out0 == true && dec_00.out1 == false && dec_00.out2 == false &&
    dec_00.out3 == false);

  // Input 01 -> output should be 0010 (only out1 = 1)
  auto dec_01 = Decoder2to4::decode(false, true);
  assert(
    dec_01.out0 == false && dec_01.out1 == true && dec_01.out2 == false &&
    dec_01.out3 == false);

  // Input 10 -> output should be 0100 (only out2 = 1)
  auto dec_10 = Decoder2to4::decode(true, false);
  assert(
    dec_10.out0 == false && dec_10.out1 == false && dec_10.out2 == true &&
    dec_10.out3 == false);
  std::cout << "2-to-4 Decoder (input: 10): ";
  std::cout << dec_10.out0 << dec_10.out1 << dec_10.out2 << dec_10.out3
            << " ✓\n";

  // Input 11 -> output should be 1000 (only out3 = 1)
  auto dec_11 = Decoder2to4::decode(true, true);
  assert(
    dec_11.out0 == false && dec_11.out1 == false && dec_11.out2 == false &&
    dec_11.out3 == true);

  // Test with enable = false -> all outputs should be 0
  auto dec_disabled = Decoder2to4::decode(true, true, false);
  assert(
    dec_disabled.out0 == false && dec_disabled.out1 == false &&
    dec_disabled.out2 == false && dec_disabled.out3 == false);

  std::cout << "All decoder tests passed! ✓\n\n";
}

int main()
{
  try
  {
    testDecoder();
  }
  catch (const std::exception &e)
  {
    std::cout << "TEST FAILED: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
