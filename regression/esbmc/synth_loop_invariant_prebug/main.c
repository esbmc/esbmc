/* The FAILED half of the pair. A claim downstream of the havoc is downgraded
 * to UNKNOWN (#7491), so the only way to pin that synthesis still *reports* a
 * genuine bug is a claim ahead of the loop. Synthesis fires and the
 * out-of-bounds write is still reported. */
int main(void)
{
  unsigned int n;
  __ESBMC_assume(n >= 1 && n <= 4);

  int a[4];
  int j = 4;
  a[j] = 1;

  unsigned int i = 0;
  unsigned int s = 0;

  while (i < n)
  {
    s = s + 2;
    i = i + 1;
  }

  return 0;
}
