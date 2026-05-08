// C++17 [expr.shift]/2: "behavior is undefined if E1 has a signed type and
// a negative value". --ub-shift-check must flag x << 1 where x = -1 under
// C++17.

int main()
{
  int x = -1;
  volatile int y = x << 1;
  (void)y;
}
