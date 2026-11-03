#include <cassert>
int nondet_int();
int main()
{
  int v = nondet_int();
  int r = 0;
  // The enclosing try routes the switch through the legacy convert_switch_case.
  try
  {
    switch (v)
    {
    case 1 ... 5:
      r = 1;
      break;
    default:
      r = 2;
    }
  }
  catch (...)
  {
  }
  assert(r == 1 || v < 1 || v > 5);
  assert(r == 2 || (v >= 1 && v <= 5));
  return 0;
}
