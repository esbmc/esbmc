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

void testComparator()
{
  std::cout << "=== Comparator Test ===\n";

  // Test equal values
  std::bitset<4> equal1("0101"), equal2("0101");
  auto comp_eq = Comparator4bit::compare(equal1, equal2);
  assert(
    comp_eq.equal == true && comp_eq.greater == false && comp_eq.less == false);

  // Test a > b
  std::bitset<4> a_greater("1010"), b_smaller("0110"); // 10 > 6
  auto comp_gt = Comparator4bit::compare(a_greater, b_smaller);
  assert(
    comp_gt.equal == false && comp_gt.greater == true && comp_gt.less == false);
  std::cout << "Compare " << a_greater << " vs " << b_smaller << ": ";
  std::cout << "Equal=" << comp_gt.equal << ", Greater=" << comp_gt.greater
            << ", Less=" << comp_gt.less << " ✓\n";

  // Test a < b
  std::bitset<4> a_smaller("0011"), b_greater("1001"); // 3 < 9
  auto comp_lt = Comparator4bit::compare(a_smaller, b_greater);
  assert(
    comp_lt.equal == false && comp_lt.greater == false && comp_lt.less == true);

  // Test edge cases
  std::bitset<4> zero("0000"), max_val("1111");
  auto comp_min_max = Comparator4bit::compare(zero, max_val);
  assert(
    comp_min_max.equal == false && comp_min_max.greater == false &&
    comp_min_max.less == true);

  auto comp_max_zero = Comparator4bit::compare(max_val, zero);
  assert(
    comp_max_zero.equal == false && comp_max_zero.greater == true &&
    comp_max_zero.less == false);

  // Test MSB difference
  std::bitset<4> msb_high("1000"), msb_low("0111"); // 8 > 7
  auto comp_msb = Comparator4bit::compare(msb_high, msb_low);
  assert(
    comp_msb.equal == false && comp_msb.greater == true &&
    comp_msb.less == false);

  std::cout << "All comparator tests passed! ✓\n\n";
}

int main()
{
  try
  {
    testComparator();
  }
  catch (const std::exception &e)
  {
    std::cout << "TEST FAILED: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
