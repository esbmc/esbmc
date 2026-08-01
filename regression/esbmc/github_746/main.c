/* THOROUGH, i.e. Linux-only: on Windows LLVM's prepare_colors() refuses colour
   for any stream that is not a console (Process::ColorNeedsFlush()), and the
   dump goes through a raw_os_ostream, so the escapes never appear there.
   github_746_nocolor pins the absence case on every platform. */
int main(void)
{
  void *p = &&lbl;   /* indirect goto: unsupported, triggers an AST dump */
lbl:
  goto *p;
  return 0;
}
