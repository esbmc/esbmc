// Second C++ translation unit. Before the ASTImporter fix, merging this TU on
// top of the first dropped every decl inside its extern "C" block, so
// multi_tu_second() resolved to a nondeterministic body and the assertion in
// main.cpp could fail.
extern "C" int multi_tu_second()
{
  return 2;
}
