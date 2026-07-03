/* CWE-835 (Loop with Unreachable Exit Condition / infinite loop).
 *
 * A bare `while (1) {}` has no reachable exit, so --termination refutes
 * the termination property via the inductive step. The non-termination
 * verdict now carries CWE-835 on a `CWE: CWE-835` line immediately after
 * the verdict comment, matching the annotation ESBMC emits for every
 * other property-violation kind.
 */
int main(void)
{
  while (1)
  {
  }
  return 0;
}
