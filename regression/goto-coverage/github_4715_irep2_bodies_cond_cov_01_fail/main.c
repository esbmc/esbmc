/* Pins --condition-coverage parity under the --irep2-bodies body round-trip.
 *
 * `(a > 0) && (b > 0)` is lowered by goto_convert into a tmp/GOTO short-circuit
 * sequence whose source location is read from the (value-level) operand
 * expression (goto_sideeffects.cpp). IREP2 value exprs carry no location, so the
 * legacy->IREP2->legacy body round-trip dropped it; condition_coverage()
 * (goto_coverage.cpp) then skipped every condition whose file is not the source
 * file, reporting "Total Conditions: 0" and a spurious VERIFICATION SUCCESSFUL
 * only under the flag. With the per-statement location re-attached to its value
 * operands the count matches the flag-off run (6 conditions) and the coverage
 * goals are reachable (VERIFICATION FAILED). A regression would flip both back
 * to 0 / SUCCESSFUL. */
#include <stdbool.h>

int main()
{
  int a, b;
  bool r = (a > 0) && (b > 0);
  return r ? 0 : 1;
}
