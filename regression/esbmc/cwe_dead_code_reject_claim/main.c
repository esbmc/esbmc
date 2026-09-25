// --dead-code-check with --claim leaves every unselected assertion SKIP but
// still in all_claims, so its probe never solves and a live branch would be
// misreported as dead (issue #4495). The combination must be rejected up front.
int main(void)
{
  return 0;
}
