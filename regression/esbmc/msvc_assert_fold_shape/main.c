/* MSVC spells assert(e) as `(void)((!!(e)) || (_wassert(...), 0))`. The
 * discarded `||` must lower to the statement it stands for, so the fold in
 * generate_ifthenelse collapses the branch into the claim glibc and Darwin
 * produce: `ASSERT g > 0`, not an `ASSERT 0` guarded by `!(g > 0)`. A claim
 * that is the constant false cannot hold on any path, which defeats every
 * consumer that asks whether it can (issue #7670). */
void _wassert(const char *_Message, const char *_File, unsigned _Line);
int g;
int main(void)
{
  (void)((!!(g > 0)) || (_wassert("g > 0", __FILE__, (unsigned)__LINE__), 0));
  return g;
}
