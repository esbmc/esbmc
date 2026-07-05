// First C++ translation unit. Its function is defined inside an extern "C"
// block, so it lives under a LinkageSpecDecl in the AST.
extern "C" int multi_tu_first()
{
  return 1;
}
