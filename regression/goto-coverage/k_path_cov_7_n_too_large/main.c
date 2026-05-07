// Regression for #4330 Copilot-3: N above the safe cap (30) must be
// rejected. Without the cap, `1 << pdepth` would overflow size_t for
// pdepth >= 64 and underflow goal-budget arithmetic for pdepth >= 62 —
// both unsound for a formal-verification tool.
int main()
{
  int a;
  return a > 0 ? 1 : 0;
}
