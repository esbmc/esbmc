// github #5868 gap 6, C++03 guard: the std::string overloads added to the
// fstream models are C++11 ([fstream.cons]) and use delegating constructors,
// which do not parse under --std c++03. No existing test included <fstream>
// under C++03, so nothing would have caught the header becoming unusable there.
#include <fstream>
#include <string>

int main()
{
  std::ifstream i("f.txt");
  i.close();
  std::ofstream o("f.txt");
  o.close();
  std::fstream f("f.txt");
  f.close();
  return 0;
}
