// Collides with the LD front end's own __ESBMC_main (built during LD's
// typecheck, before this module's final() runs). Without --secondary-entry-
// point this pins move_main_or_secondary's error path in
// clang_c_main.cpp: the collision is reported and conversion aborts.
int main(void)
{
  return 0;
}
