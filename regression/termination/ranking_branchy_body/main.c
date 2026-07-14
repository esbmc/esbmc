/* Branchy loop body (single if/else of straight-line arms) for the
 * ranking checker. From Kroening-Sharygina-Tsitovich-Wintersteiger CAV
 * 2010 Ex. The loop has the shape
 *   while (i < 255) { if (nondet) i = i + 1; else i = i + 2; }
 * which the old recognize_loop rejected outright (any GOTO/IF inside the
 * body). The extension lowers this to two paths -- {i = i + 1} and
 * {i = i + 2} -- and requires decrease on EVERY path. With m = 255 - i,
 * path 1 gives m' = m - 1 and path 2 gives m' = m - 2; both strictly
 * decrease and the bounded obligation holds from the guard.
 *
 * Expected verdict: VERIFICATION SUCCESSFUL. */

extern int __VERIFIER_nondet_int(void);

int main()
{
  int i = __VERIFIER_nondet_int();
  while (i < 255)
  {
    if (__VERIFIER_nondet_int() != 0)
      i = i + 1;
    else
      i = i + 2;
  }
  return 0;
}
