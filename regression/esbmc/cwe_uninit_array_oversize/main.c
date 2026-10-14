// Arrays above kMaxShadowedArraySize (see goto_check_uninit_vars.cpp)
// are deliberately not tracked, to avoid inflating the SMT encoding with
// a `bool[N]` shadow. The result is that uninit reads against such
// arrays go undetected — same as the pre-PR-4507 behaviour for any
// array — and verification reports SUCCESSFUL. This test pins that
// fallback: a regression here means either the cap moved or oversize
// arrays started being tracked (and inflating the encoding) again.
int main(void)
{
  int big[4097]; // one above the 4096 element cap (boundary)
  return big[0]; // uninitialised read, intentionally not flagged
}
