// Regression for GitHub #7840 (unsigned counterpart): the widened-operand
// shortcut also applies to unsigned multiplication under
// --unsigned-overflow-check — two 32-bit unsigned factors can never overflow
// a 64-bit unsigned product.
unsigned long long q(unsigned a, unsigned b)
{
  return (unsigned long long)a * (unsigned long long)b;
}
