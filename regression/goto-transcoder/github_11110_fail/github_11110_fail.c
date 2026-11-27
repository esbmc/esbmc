// #11110: array_copy into a longer heap buffer leaves its tail unconstrained.
void *malloc(__CPROVER_size_t);

int main(void)
{
  int *a = malloc(2 * sizeof(int));
  int *b = malloc(4 * sizeof(int));
  a[0] = 1;
  a[1] = 2;
  b[3] = 9;
  __CPROVER_array_copy(b, a);
  __CPROVER_assert(b[1] == 2, "copied");
  __CPROVER_assert(b[3] == 9, "tail kept");
  return 0;
}
