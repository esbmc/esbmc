// Regression for GitHub #7840 (failing counterpart): a narrowing multiply
// where both operands are typecasts, but the sources are wider than the
// destination rather than narrower — 64 + 64 > 32 — so the no-overflow
// shortcut must not fire and the genuine overflow must still be caught.
int g(long long a, long long b)
{
  return (int)a * (int)b;
}
