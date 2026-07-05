// A bare `throw;` with no exception currently being handled calls std::terminate
// ([except.throw]/9). Here nothing has been thrown, so the rethrow has no active
// exception and the program must verify FAILED. The lowering guards the rethrow
// on the exception-state typeid (0 == none): typeid 0 routes to terminate rather
// than re-raising a non-existent exception that catch(...) would swallow.
int main()
{
  try
  {
    throw; // no active exception -> std::terminate
  }
  catch (...)
  {
  }
  return 0;
}
