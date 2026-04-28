// Orthogonality check: even after the --overflow-check skip for
// non-negative-E1 lands, --ub-shift-check must still flag the
// "negative left operand" UB case in C++20 ([expr.shift]/2 explicitly
// leaves negative-E1 left-shift undefined).

int main()
{
  int x = -1;
  int y = x << 1;
  return y;
}
