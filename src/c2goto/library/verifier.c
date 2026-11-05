void __ESBMC_havoc_object(void *);
void __ESBMC_nondet_memory(void *, __SIZE_TYPE__);

void __nondet_memory_bytes(void *mem, __SIZE_TYPE__ size)
{
__ESBMC_HIDE:;
  unsigned char *p = (unsigned char *)mem;
  for (__SIZE_TYPE__ i = 0; i < size; i++)
    p[i] = __VERIFIER_nondet_uchar();
}

void __nondet_memory_impl(void *mem, __SIZE_TYPE__ size)
{
__ESBMC_HIDE:;
  // Writing the whole object in one step keeps the formula linear in its size.
  if (__ESBMC_POINTER_OFFSET(mem) == 0 && size == __builtin_object_size(mem, 0))
  {
    __ESBMC_havoc_object(mem);
    return;
  }
  __nondet_memory_bytes(mem, size);
}

void __VERIFIER_nondet_memory(void *mem, __SIZE_TYPE__ size)
{
__ESBMC_HIDE:;
  // The intrinsic havocs each target that it can prove is covered whole and
  // falls back to the functions above, which must stay linked.
  void *fallback = &__nondet_memory_impl;
  (void)fallback;
  __ESBMC_nondet_memory(mem, size);
}
