/* Negative counterpart of assigns_ptr_field_padding_pass: skipping padding in
 * the frame check must not hide a declared member missing from the clause.
 * outer's clause names the_lex and lex->depth but omits lex->token, which the
 * replaced inner may change. */
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
  __ESBMC_assigns(the_lex, lex->depth);
  inner(lex);
}

int main(void)
{
  outer(&the_lex);
  return 0;
}
