// With --result-only the success/error trace reporters bail out early, so the
// dead-store advisory must reach SARIF from a verdict-independent point.
// Pins that the CWE-563 note is emitted here even though no trace is built.
int main(void)
{
  int x = 5;
  x = 6;
  return x;
}
