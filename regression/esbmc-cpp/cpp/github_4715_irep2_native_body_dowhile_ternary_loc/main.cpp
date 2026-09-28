// convert_dowhile reads the loop-back branch's location off the condition
// operand (code.op0().find_location()). restore_value_locations stamps that
// operand with the statement's location -- but only when it has none, and
// `if2t` is the one value kind carrying its own through migrate_expr. So a
// ternary condition reports the `?` column, where the native arm substituted
// the `do` keyword's. Visible on default flags; C hides it because the frontend
// wraps a control-flow condition in a (_Bool) typecast, so the top node is not
// the ternary (W1-loc, esbmc/esbmc#4715).
bool a, b, c;

int main()
{
  do
  {
    a = true;
  } while (c ? a : b);
  return 0;
}
