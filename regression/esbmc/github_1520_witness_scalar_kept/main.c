/* Companion to github_1520_witness_no_aggregate: confirms the aggregate filter
 * does not over-prune. A plain scalar input assumption must still be emitted in
 * the GraphML witness so validators can replay the violating value. */
int main()
{
  int x;
  if (x == 5)
    assert(0);
}
