// An empty trailing catch-all leaves the try with no skip-GOTO after the pop
// (remove_unreachable prunes it), an "unsupported try-block layout" the lowering
// declines. The P4-prerequisite diagnostic reports the fallback instead of
// lowering silently; the imperative path still catches the throw, so this
// verifies SUCCESSFUL (#5075).
int main()
{
  try
  {
    throw 1;
  }
  catch (...)
  {
  }
  return 0;
}
