/* A clause's `requires` describes the state at a call site, but the enforce
 * harness starts where a global still holds its static initialiser. `other`
 * is 0 there, so `other == 1` cannot hold, the requires lowers to a false
 * assumption, and the body is never reached -- this verifies with zero VCCs
 * while breaking both its ensures and its assigns clause (#7356).
 *
 * `main` establishes the precondition, as a caller must, so
 * --replace-call-with-contract discharges the call-site assertion and does not
 * catch it either: checking the body is this mode's obligation. */
int other;
int x;
int y;

void w(int v)
{
  __ESBMC_requires(other == 1);
  __ESBMC_assigns(x);
  __ESBMC_ensures(x == v);
  x = v + 1; /* violates the ensures */
  y = 99;    /* violates the assigns clause */
}

int main()
{
  other = 1;
  w(7);
  return 0;
}
