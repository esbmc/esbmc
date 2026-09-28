// github #5868 gap 6: the fstream models declared only const char* constructors
// and open(), so the C++11 std::string overloads did not exist -- ESBMC's own
// src/python-frontend/json_utils.h hits this with `std::ifstream f(path)`.
#include <fstream>
#include <string>

int main()
{
  const std::string p = "f.txt";

  std::ofstream o(p);
  o.close();
  std::ifstream i(p);
  i.close();
  std::fstream f(p);
  f.close();

  std::ifstream j;
  j.open(p);
  j.close();
  std::ofstream k;
  k.open(p);
  k.close();
  std::fstream g;
  g.open(p);
  g.close();

  return 0;
}
