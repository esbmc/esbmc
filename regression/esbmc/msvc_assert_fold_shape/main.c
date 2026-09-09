/* MSVC spells assert(e) as `(void)((!!(e)) || (_wassert(...), 0))`. Lowering
 * the discarded `||` as a statement lets generate_ifthenelse fold it to the
 * claim glibc and Darwin produce -- `ASSERT g > 0`, not an `ASSERT 0` guarded
 * by `!(g > 0)`, which holds on no path (#7670). */
void _wassert(const char *_Message, const char *_File, unsigned _Line);
int g;
int main(void)
{
  (void)((!!(g > 0)) || (_wassert("g > 0", __FILE__, (unsigned)__LINE__), 0));
  return g;
}
