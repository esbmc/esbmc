// An empty trailing catch-all has its skip-handlers GOTO elided by the frontend
// (the jump would target the instruction right after the pop, so it is a no-op),
// leaving the pop directly followed by the handler. insert_elided_skip_gotos
// restores the canonical shape so this lowers instead of falling back to the
// imperative path (#5075). The throw is caught, so verification is SUCCESSFUL.
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
