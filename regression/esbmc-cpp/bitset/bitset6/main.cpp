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

// 4-bit Magnitude Comparator
class Comparator4bit
{
public:
  struct Result
  {
    bool equal;
    bool greater;
    bool less;
  };

  static Result compare(const std::bitset<4> &a, const std::bitset<4> &b)
  {
    bool equal = true;
    bool greater = false;

    // Compare from most significant bit to least significant bit
    for (int i = 3; i >= 0; i--)
    {
      if (a[i] != b[i])
      {
        equal = false;
        greater = a[i] && !b[i]; // a[i] = 1, b[i] = 0
        break;
      }
    }

    return {
      equal,
      greater,
      !equal && !greater // less
    };
  }
};

// Simple ALU (Arithmetic Logic Unit)
class SimpleALU
{
public:
  enum Operation
  {
    ADD = 0,
    SUB = 1,
    AND = 2,
    OR = 3
  };

  struct Result
  {
    std::bitset<4> output;
    bool carry_out;
    bool zero;
  };

  static Result
  execute(const std::bitset<4> &a, const std::bitset<4> &b, Operation op)
  {
    Result result;
    result.carry_out = false;

    switch (op)
    {
    case ADD:
    {
      auto add_result = RippleCarryAdder::add(a, b);
      result.output = add_result.sum;
      result.carry_out = add_result.carry_out;
      break;
    }
    case SUB:
    {
      // Subtraction using two's complement: a - b = a + (~b + 1)
      std::bitset<4> b_complement = ~b;
      auto sub_result = RippleCarryAdder::add(a, b_complement, true);
      result.output = sub_result.sum;
      result.carry_out = sub_result.carry_out;
      break;
    }
    case AND:
      result.output = a & b;
      break;
    case OR:
      result.output = a | b;
      break;
    }

    result.zero = (result.output == 0);
    return result;
  }
};

void testALU()
{
  std::cout << "=== ALU Test ===\n";

  // Test ADD operation
  std::bitset<4> a1("0101"), b1("0011"); // 5 + 3 = 8
  auto add_result1 = SimpleALU::execute(a1, b1, SimpleALU::ADD);
  assert(add_result1.output == std::bitset<4>("1000"));
  assert(add_result1.carry_out == false);
  assert(add_result1.zero == false);

  std::bitset<4> a2("1010"), b2("0110"); // 10 + 6 = 16 (overflow)
  auto add_result2 = SimpleALU::execute(a2, b2, SimpleALU::ADD);
  assert(add_result2.output == std::bitset<4>("0000"));
  assert(add_result2.carry_out == true);
  assert(add_result2.zero == true);
  std::cout << "ALU ADD: " << a2 << " + " << b2 << " = " << add_result2.output
            << " (carry: " << add_result2.carry_out
            << ", zero: " << add_result2.zero << ") ✓\n";

  // Test SUB operation
  std::bitset<4> a3("1000"), b3("0011"); // 8 - 3 = 5
  auto sub_result1 = SimpleALU::execute(a3, b3, SimpleALU::SUB);
  assert(sub_result1.output == std::bitset<4>("0101"));
  assert(sub_result1.zero == false);

  std::bitset<4> a4("0101"), b4("0101"); // 5 - 5 = 0
  auto sub_result2 = SimpleALU::execute(a4, b4, SimpleALU::SUB);
  assert(sub_result2.output == std::bitset<4>("0000"));
  assert(sub_result2.zero == true);

  // Test AND operation
  std::bitset<4> a5("1010"), b5("0110"); // 1010 & 0110 = 0010
  auto and_result = SimpleALU::execute(a5, b5, SimpleALU::AND);
  assert(and_result.output == std::bitset<4>("0010"));
  assert(and_result.carry_out == false);
  assert(and_result.zero == false);
  std::cout << "ALU AND: " << a5 << " & " << b5 << " = " << and_result.output
            << " (zero: " << and_result.zero << ") ✓\n";

  std::bitset<4> a6("0101"), b6("1010"); // 0101 & 1010 = 0000
  auto and_result2 = SimpleALU::execute(a6, b6, SimpleALU::AND);
  assert(and_result2.output == std::bitset<4>("0000"));
  assert(and_result2.zero == true);

  // Test OR operation
  std::bitset<4> a7("0101"), b7("1010"); // 0101 | 1010 = 1111
  auto or_result = SimpleALU::execute(a7, b7, SimpleALU::OR);
  assert(or_result.output == std::bitset<4>("1111"));
  assert(or_result.zero == false);

  std::bitset<4> a8("0000"), b8("0000"); // 0000 | 0000 = 0000
  auto or_result2 = SimpleALU::execute(a8, b8, SimpleALU::OR);
  assert(or_result2.output == std::bitset<4>("0000"));
  assert(or_result2.zero == true);

  std::cout << "All ALU tests passed! ✓\n\n";
}

int main()
{
  try
  {
    testALU();
  }
  catch (const std::exception &e)
  {
    std::cout << "TEST FAILED: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
