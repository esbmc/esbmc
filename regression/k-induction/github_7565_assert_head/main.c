/* An ASSERT loop head makes havoc_slot step back onto the entry `goto`, so that
 * one jump is both the legacy havoc slot and an entry jump. It must be havoced
 * once, not twice (#7565). */

_Bool a = 1;
_Bool b = 0;

int main()
{
  goto c;
d:
  __ESBMC_assert(b, "");
c:
  b = a;
  a = 0;
  goto d;
}
