/* github_6916_old_over_is_fresh_fail:
 * The control for github_6916_old_over_is_fresh. Materialising the snapshotted
 * value has to preserve it, not invent one, so a body that changes the field
 * the ensures pins must still be rejected. Same shape as the pass case: the
 * pointer is __ESBMC_is_fresh, so the snapshot is taken over an untyped byte
 * allocation and goes through the same path. */
typedef struct { int id; int v; } Node;

void f(Node *nodes)
{
  __ESBMC_requires(__ESBMC_is_fresh(nodes, sizeof(Node)));
  __ESBMC_assigns(nodes[0].id, nodes[0].v);
  __ESBMC_ensures(nodes[0].id == __ESBMC_old(nodes[0].id));
  nodes[0].v = 1;
  nodes[0].id = nodes[0].id + 1;   /* breaks the ensures */
}
int main(void) { return 0; }
