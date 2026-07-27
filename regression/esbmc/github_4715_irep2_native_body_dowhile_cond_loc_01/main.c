// Pins the do/while condition branch's source location under
// --irep2-native-body (W1-loc spike Phase C, esbmc/esbmc#4715).
//
// convert_dowhile reads that location off the condition *operand*
// (code.op0().find_location()), which the legacy path populates via
// restore_value_locations stamping the enclosing statement's location onto the
// round-tripped value operands. IREP2 values carry no location, so the native
// handler has to derive the same value from the statement itself; getting it
// wrong leaves the branch unlocated, which is invisible to a verdict-only test
// but breaks --condition-coverage and witness matching.
//
// The condition is side-effect-free, which is the shape this kind converts
// natively - a side-effecting one (`while (t-- > 0)`) still falls back.
int countdown(int t)
{
  int n = 0;
  do
  {
    n = n + 1;
    t = t - 1;
  } while (t > 0);
  return n;
}

int main(void)
{
  return countdown(3);
}
