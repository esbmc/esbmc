// C++20 [expr.shift]/2 (P0907R4/P1236R1): negative E1 left-shift is
// defined as wrapping (unique value congruent to E1*2^E2 mod 2^N).
// --ub-shift-check must NOT flag this under C++20.

int main()
{
  int x = -1;
  int y = x << 1;
  return y;
}
