// extern "C" keeps this main's symbol id identical to main.c's, so the C
// frontend's typecheck merges them into one shadowed definition: both
// clang_c_maint::clang_main() (for C) and clang_cpp_maint::clang_main()
// (for C++, inherited unchanged from clang_c_maint) independently attempt
// move_main_or_secondary() on it, forcing the second collision.
extern "C" int main()
{
  return 0;
}
