#include <utility>

int classify(int x)
{
  switch (x)
  {
  case 0:
    return 1;
  }
  std::unreachable();
}

int main()
{
  int x = 7;
  return classify(x);
}
