void __ESBMC_havoc_object(void *);

void __VERIFIER_nondet_memory(void *mem, __SIZE_TYPE__ size)
{
__ESBMC_HIDE:;
  // Writing the whole object in one step keeps the formula linear in its size.
  if (__ESBMC_POINTER_OFFSET(mem) == 0 && size == __builtin_object_size(mem, 0))
  {
    __ESBMC_havoc_object(mem);
    return;
  }
  unsigned char *p = (unsigned char *)mem;
  for (__SIZE_TYPE__ i = 0; i < size; i++)
    p[i] = __VERIFIER_nondet_uchar();
}
