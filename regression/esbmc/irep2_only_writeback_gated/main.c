/* The IREP2 pass changes nothing in this body: migrate_expr already decayed
   the array in the conditional arm, and the call's argument needs no
   conversion the converter did not emit. adjust() therefore leaves the legacy
   value alone, and --symbol-table-only prints the converter's tree unless the
   write-back is forced. */
char b[4];
char *d;
void snk(void *p);

int main(int argc, char **argv)
{
  char *c;
  c = argc == 1 ? b : d;
  snk(c);
  return 0;
}
