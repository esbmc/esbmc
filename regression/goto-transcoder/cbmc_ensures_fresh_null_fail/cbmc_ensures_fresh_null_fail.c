int *alloc_one(void)
  __CPROVER_ensures(__CPROVER_is_fresh(__CPROVER_return_value, sizeof(int)))
{
  return 0; /* NULL: certainly not fresh */
}
int main() { return 0; }
