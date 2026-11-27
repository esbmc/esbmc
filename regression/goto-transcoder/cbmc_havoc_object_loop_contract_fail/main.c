/* Built with goto-instrument --apply-loop-contracts, which havocs the loop's
 * assigned heap buffer with __CPROVER_havoc_object. */
void *malloc(__CPROVER_size_t);

int main(void)
{
  unsigned char *buf = malloc(8);
  unsigned char *other = malloc(1);
  *other = 7;
  unsigned i = 0;
  while (i < 8)
    __CPROVER_assigns(i, __CPROVER_object_whole(buf))
    __CPROVER_loop_invariant(i <= 8)
  {
    buf[i] = 0;
    i++;
  }
  // The invariant says nothing about the havocked buffer.
  __CPROVER_assert(buf[0] == 0, "buf[0] cleared");
  return 0;
}
