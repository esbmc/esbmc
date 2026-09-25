/* clang_c_main looks up argc'/argv' unconditionally when main takes arguments;
   the symbols are a side effect of the adjust pass, so the sole adjuster owes
   them or the lookup dereferences null. */
int main(int argc, char **argv)
{
  return argc > 0 ? 0 : (int)argv[0][0];
}
