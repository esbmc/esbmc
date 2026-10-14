/* github_6916_old_over_is_fresh:
 * __ESBMC_old takes the address of its operand so that the pre-state value can
 * be read back through it. That needs an lvalue. __ESBMC_is_fresh backs the
 * object with an untyped byte array, so reading n->id through it lowers to a
 * value reassembled from bytes, and its address reached the solver as
 * address_of(bitcast(concat(...))), which convert_addr_of aborts on. */
typedef struct { int id; int v; } Node;

void helper(Node *n) {
  __ESBMC_assigns(n->v);
  __ESBMC_ensures(n->id == __ESBMC_old(n->id));
  n->v = 1;
}

void f(Node *nodes) {
  __ESBMC_requires(__ESBMC_is_fresh(nodes, sizeof(Node)));
  __ESBMC_assigns(nodes[0].v);
  __ESBMC_ensures(1);
  helper(&nodes[0]);
}
int main(void) { return 0; }
