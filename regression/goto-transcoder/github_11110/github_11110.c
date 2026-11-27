// #11110: __CPROVER_array_copy/_array_replace between heap buffers.
void *malloc(__CPROVER_size_t);

int main(void)
{
  int *a = malloc(4 * sizeof(int));
  int *b = malloc(4 * sizeof(int));
  a[2] = 7;
  __CPROVER_array_copy(b, a);
  __CPROVER_assert(b[2] == 7, "copied");

  int *c = malloc(2 * sizeof(int));
  c[0] = 1;
  c[1] = 2;
  b[0] = 5;
  b[3] = 9;
  __CPROVER_array_replace(b + 1, c);
  __CPROVER_assert(b[0] == 5 && b[1] == 1 && b[2] == 2, "replaced");
  __CPROVER_assert(b[3] == 9, "tail kept");
  return 0;
}
