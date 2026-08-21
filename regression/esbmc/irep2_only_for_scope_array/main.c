/* The loop variable's DEAD must come after the loop that reads it; the
   for-init hoist is what puts it there. See docs/roadmap/scope-clang-c-irep2.md
   §100. */
int nondet_int(void);
char buf[11];

int main(void)
{
  for (int i = 0; i < 10; ++i)
    buf[i] = (char)nondet_int();

  buf[10] = 0;
  return buf[0];
}
