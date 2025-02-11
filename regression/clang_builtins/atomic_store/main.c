int main() {
  int v = 0;
  int count = 0;

  __atomic_store_n(&v, count + 1, __ATOMIC_RELAXED);
  assert(v == ++count);

  __atomic_store_n(&v, count + 1, __ATOMIC_RELEASE);
  assert(v == ++count);

  __atomic_store_n(&v, count + 1, __ATOMIC_SEQ_CST);
  assert(v == ++count);

  /* Now test the generic variant.  */
  count++;

  __atomic_store(&v, &count, __ATOMIC_RELAXED);
  assert(v == count++);

  __atomic_store(&v, &count, __ATOMIC_RELEASE);
  assert(v == count++);

  __atomic_store(&v, &count, __ATOMIC_SEQ_CST);
  assert(v == count);

  return 0;
}
