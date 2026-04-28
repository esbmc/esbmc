// Regression test for ESBMC issue #3711.
// Original example contributed by MarkAA-uw.

int mk(const int ofs)
{
  int res;
  res = res < 0 ? -res : res;
  res = (res % 128) * 2 + ofs;
  return res;
}

int in_range(const int n)
{
  return 0 <= n && n < 256;
}

int main()
{
  int qn;
  int qm;

  //--------------------------------------------------
  // Originally FAILED with function call in the middle
  qn = mk(0);
  qm = mk(1);

  __ESBMC_loop_invariant(
    0 <= qn && qn < 256 &&
    in_range(qm) &&
    (qn != qm));
  for(int count = 0; count < 10; count++)
  {
    qn = (qn + 2) % 256;
    qm = (qm + 2) % 256;
  }

  //--------------------------------------------------
  // PASSES: function call at the end
  qn = mk(0);
  qm = mk(1);

  __ESBMC_loop_invariant(
    0 <= qn && qn < 256 &&
    (qn != qm) &&
    in_range(qm));
  for(int count = 0; count < 10; count++)
  {
    qn = (qn + 2) % 256;
    qm = (qm + 2) % 256;
  }

  //--------------------------------------------------
  // PASSES: function call at the beginning
  qn = mk(0);
  qm = mk(1);

  __ESBMC_loop_invariant(
    in_range(qm) &&
    0 <= qn && qn < 256 &&
    (qn != qm));
  for(int count = 0; count < 10; count++)
  {
    qn = (qn + 2) % 256;
    qm = (qm + 2) % 256;
  }

  //--------------------------------------------------
  // PASSES: no function call in invariant
  qn = mk(0);
  qm = mk(1);

  __ESBMC_loop_invariant(
    0 <= qn && qn < 256 &&
    0 <= qm && qm < 256 &&
    (qn != qm));
  for(int count = 0; count < 10; count++)
  {
    qn = (qn + 2) % 256;
    qm = (qm + 2) % 256;
  }

  return 0;
}

