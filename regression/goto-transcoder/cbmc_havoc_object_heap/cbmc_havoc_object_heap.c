void *malloc(__CPROVER_size_t);
void __CPROVER_havoc_object(void *);

int main(void)
{
  int *p = malloc(3 * sizeof(int));
  p[0] = 1;
  // A pointer into the object, not to its start: CBMC still havocs it whole.
  __CPROVER_havoc_object(p + 1);
  __CPROVER_assert(p[0] == 1, "p[0] kept");
  return 0;
}
