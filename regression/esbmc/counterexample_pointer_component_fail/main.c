// A pointer element the model resolves is reported as itself: an address and a
// null pointer are values the model produced, not expressions symex propagated.
int x = 5;
int *arr[2];

int main(void)
{
  arr[0] = &x;
  arr[1] = 0;
  __ESBMC_assert(arr[0] != &x, "arr0 is &x");
  return 0;
}
