static int shared;
int main()
{
  __CPROVER_assert(__CPROVER_OBJECT_SIZE(&shared) == 4, "scalar static is 4 bytes");
  return shared;
}
