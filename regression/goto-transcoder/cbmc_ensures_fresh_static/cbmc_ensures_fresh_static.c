static int shared;

int *alloc_one(void)
  __CPROVER_ensures(__CPROVER_is_fresh(__CPROVER_return_value, sizeof(int)))
{
  return &shared; /* not fresh: a static, returned on every call */
}

int main() { return 0; }
