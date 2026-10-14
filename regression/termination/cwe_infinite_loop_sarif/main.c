/* CWE-835 in the SARIF output for a non-termination verdict.
 *
 * A non-termination verdict has no counterexample trace (it is proven by
 * UNSAT at the loop exit marker), so ESBMC anchors a synthetic single-step
 * trace to the loop's marker to drive the SARIF emitter. The report must
 * carry CWE-835 both as a result-level `taxa` reference and in the
 * run-level `taxonomies` block.
 */
int main(void)
{
  while (1)
  {
  }
  return 0;
}
