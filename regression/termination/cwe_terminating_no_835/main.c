/* Negative case for CWE-835: a loop that provably terminates must NOT be
 * flagged as an infinite loop. A linear ranking function bounds the loop,
 * so --termination proves the termination property and reports
 * VERIFICATION SUCCESSFUL with no CWE annotation.
 */
int main(void)
{
  int i = 0;
  while (i < 3)
    i++;
  return 0;
}
