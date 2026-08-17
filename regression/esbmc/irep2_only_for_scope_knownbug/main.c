/* The loop variable's DEAD must come after the loop that reads it. Under
   --clang-c-irep2-adjust-only it is emitted before the body -- `i` is marked
   dead while still live. See docs/roadmap/scope-clang-c-irep2.md §98. */
int nondet_int(void);
char buf[11];

int main(void)
{
  for (int i = 0; i < 10; ++i)
    buf[i] = (char)nondet_int();

  buf[10] = 0;
  return buf[0];
}
