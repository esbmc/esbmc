// Interval analysis runs before the overflow checks are added, so folding
// a + 1 into a constant used to drop its overflow claim.
int main()
{
  int a = 2147483647;
  int b = a + 1;
  return b;
}
