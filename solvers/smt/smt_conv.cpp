#include "smt_conv.h"

smt_ast::smt_ast(const smt_sort *s, smt_func_kind k, const smt_ast *a)
{
  smt_ast(s, k, a, NULL, NULL);
}

smt_ast::smt_ast(const smt_sort *s, smt_func_kind k, const smt_ast *a, const smt_ast *b)
{
  smt_ast(s, k, a, b, NULL);
}

smt_ast::smt_ast(const smt_sort *s, smt_func_kind k, const smt_ast *a,
                 const smt_ast *b, const smt_ast *c)
{
  sort = s;
  kind = k;
  arguments[0] = a;
  arguments[1] = b;
  arguments[2] = c;
}

smt_ast::~smt_ast(void)
{
  if (arguments[0])
    delete arguments[0];
  if (arguments[1])
    delete arguments[1];
  if (arguments[2])
    delete arguments[2];
}

smt_convt::smt_convt(void)
{
}

smt_convt::~smt_convt(void)
{
}

void
smt_convt::push_ctx(void)
{
  prop_convt::push_ctx();
}

void
smt_convt::pop_ctx(void)
{
  prop_convt::pop_ctx();
}
