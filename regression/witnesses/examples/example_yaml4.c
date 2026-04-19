void reach_error() {}

struct A
{
  int x;
};
struct B
{
  struct A a;
};

int main()
{
  struct B b = {{0}};
  if (b.a.x == 0)
    reach_error();
}
