// An empty trailing catch-all leaves the try with no skip-GOTO after the pop:
// the empty handler's entry coincides with the after-try point, so normal
// completion just falls through it. normalize_empty_handlers synthesizes the
// missing skip-GOTO so the lowering handles this in-line rather than falling
// back to the imperative path. The catch-all swallows the throw -> SUCCESSFUL.
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
