// No other language module is present, so clang_c_maint::clang_main()'s
// move of __ESBMC_main into context cannot collide: this pins the
// move_main_or_secondary no-collision path (clang_c_main.cpp) independently
// of the LD front end, which is what every other test in this directory
// exercises together with it.
int main(void)
{
  return 0;
}
