/* A pointer into the middle of an object is dynamic iff the object is. */
void *malloc(__SIZE_TYPE__);

struct pair
{
  int a;
  int b;
};

int main(void)
{
  unsigned i;
  __CPROVER_assume(i < 4);
  int *arr = malloc(4 * sizeof(int));
  struct pair *s = malloc(sizeof(struct pair));
  __CPROVER_assume(arr != 0 && s != 0);
  __CPROVER_assert(__CPROVER_DYNAMIC_OBJECT(arr + i), "array element");
  __CPROVER_assert(__CPROVER_DYNAMIC_OBJECT(&s->b), "struct member");
  int stack[4];
  __CPROVER_assert(!__CPROVER_DYNAMIC_OBJECT(stack + i), "stack element");
  return 0;
}
