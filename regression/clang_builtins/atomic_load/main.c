int main() {
  int v = 0;
  int count = 0;

  assert(__atomic_load_n(&v, __ATOMIC_RELAXED) == count++);
  v++;

  assert(__atomic_load_n(&v, __ATOMIC_ACQUIRE) == count++);
  v++;

  assert(__atomic_load_n(&v, __ATOMIC_CONSUME) == count++);
  v++;

  assert(__atomic_load_n(&v, __ATOMIC_SEQ_CST) == count++);
  v++;

  /* Now test the generic variants.  */

  __atomic_load(&v, &count, __ATOMIC_RELAXED);
  assert(count == v);
  v++;

  __atomic_load(&v, &count, __ATOMIC_ACQUIRE);
  assert(count == v);
  v++;

  __atomic_load(&v, &count, __ATOMIC_CONSUME);
  assert(count == v);
  v++;

  __atomic_load(&v, &count, __ATOMIC_SEQ_CST);
  assert(count == v);
  v++;

  return 0;
}
