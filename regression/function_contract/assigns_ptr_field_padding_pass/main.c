/* The frame check of an enforced function snapshots every member of a struct
 * reached through a pointer parameter. Padding members carry no value, and a
 * replaced callee that havocs the whole struct changes them, so they must not
 * be snapshotted: the assigns clause below covers every declared member. */
typedef struct
{
  unsigned long depth;
  int token; /* followed by 4 bytes of padding */
} lex_t;

lex_t the_lex;

void inner(lex_t *lex)
{
  __ESBMC_ensures(lex->depth == __ESBMC_old(lex->depth));
  __ESBMC_assigns(the_lex, lex->depth, lex->token);
  lex->token = 1;
}

void outer(lex_t *lex)
{
  __ESBMC_requires(lex == &the_lex);
  __ESBMC_ensures(lex->depth == __ESBMC_old(lex->depth));
  __ESBMC_assigns(the_lex, lex->depth, lex->token);
  inner(lex);
}

int main(void)
{
  outer(&the_lex);
  return 0;
}
