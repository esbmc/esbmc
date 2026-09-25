/* The sharper half: the body writes a valid element, just not the one the
 * clause names. Only an encoding that tracks *which* index was spared reports
 * this -- one that merely notices "an element changed" would not. */
int global[1000];

void f(int i, int j, int v)
{
  __ESBMC_requires(i >= 0 && i < 1000);
  __ESBMC_requires(j >= 0 && j < 1000);
  __ESBMC_assigns(global[i]);
  __ESBMC_ensures(1);
  global[j] = v; /* j need not equal i */
}

int main()
{
  return 0;
}
