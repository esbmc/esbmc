/* Terminating companion to github_4426_infinite_loop (esbmc/esbmc#4426).
 *
 * Same harness shape (main -> ldv_stop -> loop), but the loop is bounded and
 * therefore terminates. The result must stay VERIFICATION SUCCESSFUL: the
 * #4426 fix (not rewriting bare self-loops away under --termination) must not
 * make a genuinely terminating loop report non-termination.
 */
void ldv_stop(int n)
{
  int i = 0;
  while (i < n)
    i++;
}

int main(void)
{
  ldv_stop(3);
  return 0;
}
