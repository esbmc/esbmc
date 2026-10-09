// C's clang_main() collides with LD's __ESBMC_main first and claims the
// __ESBMC_secondary_main slot (the success half of move_main_or_secondary's
// rename branch, clang_c_main.cpp). main.cpp's extern "C" main shares this
// same symbol id, so C++'s own clang_main() then collides a second time and
// finds __ESBMC_secondary_main already taken too.
int main(void)
{
  return 0;
}
