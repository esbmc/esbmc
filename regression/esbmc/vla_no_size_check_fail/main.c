// vla_no_size_check's program with the check left on: 1000^3 ints cannot be
// laid out under a 32-bit PTRDIFF_MAX, so the declaration is reported. Pins
// that --no-vla-size-check is what suppresses it, not the bound being gone
// (#7306).
int main(void)
{
  int n = 1000;
  int a[n][n][n];
  a[0][0][0] = 1;
  return a[0][0][0] - 1;
}
