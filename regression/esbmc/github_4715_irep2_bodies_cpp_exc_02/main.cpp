// Multi-handler variant of github_4715_irep2_bodies_cpp_exc_01 (V.4.3,
// esbmc#4715). Two catch clauses force the cpp-catch operand list to carry
// more than one handler, so the back-migration must re-attach each handler's
// exception_id from exception_list[i-1] for i > 1 (a single-handler test never
// exercises that index). A char is thrown, the second handler (catch(char))
// matches, and its in-handler assertion holds, so verification must SUCCEED.
int main()
{
  try
  {
    throw 'c';
    __ESBMC_assert(0, "unreachable after throw");
  }
  catch (int e)
  {
    __ESBMC_assert(0, "int handler must not catch a char");
  }
  catch (char e)
  {
    __ESBMC_assert(e == 'c', "caught value is 'c'");
  }
  return 0;
}
