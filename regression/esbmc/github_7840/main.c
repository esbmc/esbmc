// Regression for GitHub #7840: --overflow-check on a multiplication whose
// result is __int128, with both factors widened from long long, used to blow
// up the SMT encoding (a 256-bit multiplier UNSAT proof) badly enough to OOM
// or time out across three independent solver backends for a single VCC.
// Two 64-bit signed factors can never overflow a 128-bit signed product, so
// this is provably safe and must resolve without ever reaching the solver.
long long f(long long T, long long s)
{
  __int128 result = (__int128)T * (__int128)s;
  return (long long)result;
}
