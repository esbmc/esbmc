/* Faithful reproducer from esbmc/esbmc discussion #5701. */
int main()
{
  bool short_radar;
  bool lo_speed;
  bool rl_mode = 0;
  bool req_mode;

  int loop_idx = 0;
  while (true)
  {
    req_mode = nondet_bool();
    if (lo_speed)
      if (req_mode)
      {
        rl_mode = true;
      }
      else
      {
        rl_mode = false;
        __ESBMC_assert(rl_mode == false, "rl_mode asn");
      }
    else
      switch (rl_mode)
      {
      case false:
        rl_mode = false;
        break;
      default:
        break;
      }
    __ESBMC_assert(!(rl_mode == false) || (lo_speed && short_radar), "park");
    ++loop_idx;
  }
}
