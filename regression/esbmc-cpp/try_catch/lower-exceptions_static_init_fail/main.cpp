// A global object whose constructor throws: the exception escapes during static
// initialisation (before main) and is uncaught -> std::terminate. main's own
// function-try-block cannot catch it.
struct E
{
  int v;
  E(int a) : v(a)
  {
  }
};
struct C
{
  C()
  {
    throw E(1);
  }
};

C global_c; // constructed before main; its throw is uncaught

int main()
{
  return 0;
}
