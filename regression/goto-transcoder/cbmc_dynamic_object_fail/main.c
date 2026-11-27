/* p is the heap object on one path and a stack object on the other, so the
 * assertion fails only on the heap path. */
void *malloc(__SIZE_TYPE__);

int main(void)
{
  int local;
  _Bool pick_heap;
  int *heap = malloc(sizeof(int));
  __CPROVER_assume(heap != 0);
  int *p = pick_heap ? heap : &local;
  __CPROVER_assert(!__CPROVER_DYNAMIC_OBJECT(p), "p is not dynamic");
  return 0;
}
