// C++20 [expr.shift]/2: negative E1 left-shift is defined as wrapping.
// --ub-shift-check must NOT flag x << 1 where x = -1 under C++20.

int main()
{
  int x = -1;
  volatile int y = x << 1;
  (void)y;
}
