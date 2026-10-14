// A three-dimensional VLA is larger than PTRDIFF_MAX on a 32-bit target --
// 1000^3 ints is 4 GB against a 2 GB cap -- so the PTRDIFF_MAX bound reports
// the declaration. --no-vla-size-check is "do not check whether the size of
// VLAs overflows the available address space", which is exactly this check, so
// the bound has to honour it. SV-COMP passes the flag on every task and this
// shape (sv-benchmarks c/array-multidimensional/init-3-u.c) turned 17 correct
// verdicts into wrong ones without it (#7306).
//
// See vla_no_size_check_fail for the same program with the check left on.
int main(void)
{
  int n = 1000;
  int a[n][n][n];
  a[0][0][0] = 1;
  return a[0][0][0] - 1;
}
