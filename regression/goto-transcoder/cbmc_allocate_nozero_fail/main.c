/* With the flag clear the object is not zero-filled, so its contents are
 * nondet. */
void *__CPROVER_allocate(__CPROVER_size_t, __CPROVER_bool);
int main(void)
{
  int *p = __CPROVER_allocate(4 * sizeof(int), 0);
  __CPROVER_assert(p[3] == 0, "the object need not be zero-filled");
  return 0;
}
