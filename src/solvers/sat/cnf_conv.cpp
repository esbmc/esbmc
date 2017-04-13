#include <cnf_conv.h>

cnf_convt::cnf_convt(cnf_iface *_cnf_api)
  : sat_iface(), cnf_api(_cnf_api)
{
}

cnf_convt::~cnf_convt()
{
}

literalt
cnf_convt::lnot(literalt a)
{
  a.invert();
  return a;
}

literalt
cnf_convt::lselect(literalt a, literalt b, literalt c)
{  // a?b:c = (a AND b) OR (/a AND c)
  if(a==const_literal(true)) return b;
  if(a==const_literal(false)) return c;
  if(b==c) return b;

  bvt bv;
  bv.reserve(2);
  literalt one = land(a, b);
  literalt two = land(lnot(a), c);
  return lor(one, two);
}

literalt
cnf_convt::lequal(literalt a, literalt b)
{
  return lnot(lxor(a, b));
}

literalt
cnf_convt::limplies(literalt a, literalt b)
{
  return lor(lnot(a), b);
}

literalt
cnf_convt::lxor(literalt a, literalt b)
{
  if (a == const_literal(false)) return b;
  if (b == const_literal(false)) return a;
  if (a == const_literal(true)) return lnot(b);
  if (b == const_literal(true)) return lnot(a);

  literalt output = this->new_variable();
  gate_xor(a, b, output);
  return output;
}

literalt
cnf_convt::lor(literalt a, literalt b)
{
  if (a == const_literal(false)) return b;
  if (b == const_literal(false)) return a;
  if (a == const_literal(true)) return const_literal(true);
  if (b == const_literal(true)) return const_literal(true);

  literalt output = this->new_variable();
  gate_or(a, b, output);
  return output;
}

literalt
cnf_convt::land(literalt a, literalt b)
{
  if (a == const_literal(true)) return b;
  if (b == const_literal(true)) return a;
  if (a == const_literal(false)) return const_literal(false);
  if (b == const_literal(false)) return const_literal(false);
  if (a == b) return a;

  literalt output = this->new_variable();
  gate_and(a, b, output);
  return output;
}

void
cnf_convt::gate_xor(literalt a, literalt b, literalt o)
{
  // a xor b = o <==> (a' + b' + o')
  //                  (a + b + o' )
  //                  (a' + b + o)
  //                  (a + b' + o)
  bvt lits;

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(neg(b));
  lits.push_back(neg(o));
  cnf_api->lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  cnf_api->lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(pos(b));
  lits.push_back(pos(o));
  cnf_api->lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  cnf_api->lcnf(lits);
}

void
cnf_convt::gate_or(literalt a, literalt b, literalt o)
{
  // a+b=c <==> (a' + c)( b' + c)(a + b + c')
  bvt lits;

  lits.clear();
  lits.reserve(2);
  lits.push_back(neg(a));
  lits.push_back(pos(o));
  cnf_api->lcnf(lits);

  lits.clear();
  lits.reserve(2);
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  cnf_api->lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  cnf_api->lcnf(lits);
}

void
cnf_convt::gate_and(literalt a, literalt b, literalt o)
{
  // a*b=c <==> (a + o')( b + o')(a'+b'+o)
  bvt lits;

  lits.clear();
  lits.reserve(2);
  lits.push_back(pos(a));
  lits.push_back(neg(o));
  cnf_api->lcnf(lits);

  lits.clear();
  lits.reserve(2);
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  cnf_api->lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  cnf_api->lcnf(lits);
}

void
cnf_convt::set_equal(literalt a, literalt b)
{
  if (a == const_literal(false)) {
    cnf_api->setto(b, false);
    return;
  } else if (b == const_literal(false)) {
    cnf_api->setto(a, false);
    return;
  } else if (a == const_literal(true)) {
    cnf_api->setto(b, true);
    return;
  } else if (b == const_literal(true)) {
    cnf_api->setto(a, true);
    return;
  }

  bvt bv;
  bv.resize(2);
  bv[0] = a;
  bv[1] = lnot(b);
  cnf_api->lcnf(bv);

  bv[0] = lnot(a);
  bv[1] = b;
  cnf_api->lcnf(bv);
  return;
}
