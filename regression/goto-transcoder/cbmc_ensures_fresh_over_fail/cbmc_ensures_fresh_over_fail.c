static int shared;

int *alloc_one(void)
  __CPROVER_ensures(__CPROVER_is_fresh(__CPROVER_return_value, 1000))
{
  return &shared; /* only 4 bytes, but the contract promises 1000 */
}
int main() { return 0; }
