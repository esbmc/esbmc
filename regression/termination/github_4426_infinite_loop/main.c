/* Regression for esbmc/esbmc#4426
 * (ldv-linux-4.0-rc1-mav/linux-4.0-rc1---drivers--media--rc--lirc_dev.ko.cil).
 *
 * The LDV harness ends every path in ldv_stop(), which is a bare unconditional
 * infinite loop. The program therefore never terminates, so the termination
 * property is FALSE.
 *
 * A bare self-loop `A: goto A;` used to be rewritten to assume(false) (in
 * goto_loopst::find_function_loops and again in symex_goto), which is sound for
 * reachability but erases the non-termination under --termination, yielding a
 * spurious VERIFICATION SUCCESSFUL (0 VCCs, "forward condition shows all
 * executions terminate"). Both rewrites are now skipped under --termination.
 */
void ldv_stop(void)
{
  ldv_stop_label:;
  goto ldv_stop_label;
}

int main(void)
{
  ldv_stop();
  return 0;
}
