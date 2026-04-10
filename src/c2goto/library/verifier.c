void __VERIFIER_nondet_memory(void *mem, __SIZE_TYPE__ size)
{
  unsigned char *p = (unsigned char *)mem;
  for (__SIZE_TYPE__ i = 0; i < size; i++)
    p[i] = __VERIFIER_nondet_uchar();
}