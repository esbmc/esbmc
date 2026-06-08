// C++20 [expr.shift]/2: signed left-shift is defined as wrapping for any
// value of E1. --overflow-check must NOT flag this under C++20.

extern "C" int nondet_int();

int main()
{
  int x = nondet_int();
  int y = x << 1;
  return y;
}
