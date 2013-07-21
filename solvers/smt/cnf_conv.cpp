// Danger Will Robinson: this is not a C++ class, but in fact a template, and
// is included by cnf_conv.h directly so that uses of it are instanciated
// correctly.

template <class subclass>
cnf_convt<subclass>::cnf_convt(bool enable_cache, bool int_encoding,
                      const namespacet &_ns, bool is_cpp, bool tuple_support,
                      bool bools_in_arrs, bool can_init_inf_arrs)
  : subclass(enable_cache, int_encoding, _ns, is_cpp, tuple_support,
              bools_in_arrs, can_init_inf_arrs)
{
}

template <class subclass>
cnf_convt<subclass>::~cnf_convt()
{
}

template <class subclass>
literalt
cnf_convt<subclass>::lnot(literalt a)
{
  a.invert();
  return a;
}

template <class subclass>
literalt
cnf_convt<subclass>::lselect(literalt a, literalt b, literalt c)
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

template <class subclass>
literalt
cnf_convt<subclass>::lequal(literalt a, literalt b)
{
  return lnot(lxor(a, b));
}

template <class subclass>
literalt
cnf_convt<subclass>::limplies(literalt a, literalt b)
{
  return lor(lnot(a), b);
}

template <class subclass>
literalt
cnf_convt<subclass>::lxor(literalt a, literalt b)
{
  if (a == const_literal(false)) return b;
  if (b == const_literal(false)) return a;
  if (a == const_literal(true)) return lnot(b);
  if (b == const_literal(true)) return lnot(a);

  literalt output = this->new_variable();
  gate_xor(a, b, output);
  return output;
}

template <class subclass>
literalt
cnf_convt<subclass>::lor(literalt a, literalt b)
{
  if (a == const_literal(false)) return b;
  if (b == const_literal(false)) return a;
  if (a == const_literal(true)) return const_literal(true);
  if (b == const_literal(true)) return const_literal(true);

  literalt output = this->new_variable();
  gate_or(a, b, output);
  return output;
}

template <class subclass>
literalt
cnf_convt<subclass>::land(literalt a, literalt b)
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

template <class subclass>
void
cnf_convt<subclass>::gate_xor(literalt a, literalt b, literalt o)
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
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(pos(b));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  lcnf(lits);
}

template <class subclass>
void
cnf_convt<subclass>::gate_or(literalt a, literalt b, literalt o)
{
  // a+b=c <==> (a' + c)( b' + c)(a + b + c')
  bvt lits;

  lits.clear();
  lits.reserve(2);
  lits.push_back(neg(a));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(2);
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  lcnf(lits);
}

template <class subclass>
void
cnf_convt<subclass>::gate_and(literalt a, literalt b, literalt o)
{
  // a*b=c <==> (a + o')( b + o')(a'+b'+o)
  bvt lits;

  lits.clear();
  lits.reserve(2);
  lits.push_back(pos(a));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(2);
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  lcnf(lits);
}

template <class subclass>
void
cnf_convt<subclass>::set_equal(literalt a, literalt b)
{
  if (a == const_literal(false)) {
    setto(b, false);
    return;
  } else if (b == const_literal(false)) {
    setto(a, false);
    return;
  } else if (a == const_literal(true)) {
    setto(b, true);
    return;
  } else if (b == const_literal(true)) {
    setto(a, true);
    return;
  }

  bvt bv;
  bv.resize(2);
  bv[0] = a;
  bv[1] = lnot(b);
  lcnf(bv);

  bv[0] = lnot(a);
  bv[1] = b;
  lcnf(bv);
  return;
}
