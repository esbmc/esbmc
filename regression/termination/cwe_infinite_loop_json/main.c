/* CWE-835 in the JSON report for a non-termination verdict.
 *
 * The synthetic trace that carries the non-termination verdict must reach
 * the --generate-json-report channel too: report.json records the verdict
 * as a violation whose assertion.cwe array contains 835.
 */
int main(void)
{
  while (1)
  {
  }
  return 0;
}
