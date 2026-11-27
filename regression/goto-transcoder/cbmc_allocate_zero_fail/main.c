void *__CPROVER_allocate(__CPROVER_size_t, __CPROVER_bool);
int main(void)
{
  int *p = __CPROVER_allocate(4 * sizeof(int), 1);
  __CPROVER_assert(p[3] == 1, "a zero-filled object does not hold 1");
  return 0;
}
