/* __CPROVER_DYNAMIC_OBJECT(p) holds iff p is a heap pointer; a malloc that
 * returns NULL must not make the null pointer dynamic. */
void *malloc(__SIZE_TYPE__);

int global;

int main(void)
{
  int local;
  int *heap = malloc(sizeof(int));
  if (heap)
    __CPROVER_assert(__CPROVER_DYNAMIC_OBJECT(heap), "heap is dynamic");
  else
    __CPROVER_assert(!__CPROVER_DYNAMIC_OBJECT(heap), "null is not dynamic");
  __CPROVER_assert(!__CPROVER_DYNAMIC_OBJECT(&local), "local is not dynamic");
  __CPROVER_assert(!__CPROVER_DYNAMIC_OBJECT(&global), "global is not dynamic");
  return 0;
}
