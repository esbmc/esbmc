// Regression test for ESBMC issue #3711 (_fail1).
// Based on original example contributed by MarkAA-uw.
// This variant has an intentionally wrong loop invariant (base case fails).

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
  // WRONG: invariant contradicts mk(), should fail
  // (qn is always >= 0, so qn < 0 is false at loop entry)
  qn = mk(0);
  qm = mk(1);

  __ESBMC_loop_invariant(
    0 <= qn && qn < 256 &&
    in_range(qm) &&
    qn < 0);
  for(int count = 0; count < 10; count++)
  {
    qn = (qn + 2) % 256;
    qm = (qm + 2) % 256;
  }

  return 0;
}

