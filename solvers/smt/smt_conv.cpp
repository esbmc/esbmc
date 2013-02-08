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
}
