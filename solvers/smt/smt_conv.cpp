#include "smt_conv.h"

smt_sort::smt_sort(smt_sort_kind k)
{
  kind = k;
}

smt_sort::~smt_sort(void)
{
}

smt_ast::smt_ast(const smt_sort *s, smt_func_kind k)
{
  sort = s;
  kind = k;
}

smt_ast::~smt_ast(void)
{
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
