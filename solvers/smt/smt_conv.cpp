#include "smt_conv.h"

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

  union_varst::nth_index<1>::type &union_numindex = union_vars.get<1>();
  union_numindex.erase(ctx_level);
    union_numindex.erase(ctx_level);
}
