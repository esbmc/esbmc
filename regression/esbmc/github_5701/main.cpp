/* Deterministic reproducer for esbmc/esbmc discussion #5701.
 *
 * lo_speed is pinned false, so the `if (lo_speed)` body -- including the inner
 * `else` assignment `rl_mode = false;` -- can never execute.  Yet
 * under `--no-slice` (which keeps internal/hidden SSA steps) the hidden
 * phi-merge node for rl_mode used to be printed in the counterexample at that
 * assignment line, a "surprising state" for a line that never ran.  The
 * human-readable trace must contain NO state at that line, regardless of the
 * solver. */
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
    __ESBMC_assume(!lo_speed);
    if (lo_speed)
      if (req_mode)
        rl_mode = true;
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
