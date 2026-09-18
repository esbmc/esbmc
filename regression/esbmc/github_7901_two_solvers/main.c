/* esbmc/esbmc#7901: two solver flags at once is a rejected input, not a crash.
 * Both flags exist in every build; only availability varies. */
int main(void)
{
  return 0;
}
