/* As github_10939_ir_field_check, but the union's full-size member over-reads
 * the source's first field, reported as for a struct target (#10888). */
struct d { unsigned char a; unsigned char b; unsigned short c; };
union u { unsigned w; };
int main(void)
{
  struct d d = {1, 2, 3};
  union u v = *(union u *)&d;
  return v.w;
}
