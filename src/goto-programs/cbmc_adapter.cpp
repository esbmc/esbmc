// Faithful C++ port of goto-transcoder's adapter.rs: rewrites CBMC irep
// conventions into ESBMC's so the result feeds symbolt::from_irep and the
// goto_program_irep convert() directly. Function/struct names and control flow
// mirror the Rust to keep the two implementations easy to diff.

#include <goto-programs/cbmc_adapter.h>
#include <util/c_types.h>
#include <util/message.h>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace
{
// Construct an id-only irep (mirrors Rust Irept::from(<string>)).
irept mk(const irep_idt &id)
{
  return irept(id);
}

// Mirrors Rust's `named_subt.contains_key`, honouring ESBMC's '#'-comment
// routing (all keys used here are non-'#', i.e. true named-subs).
bool has_sub(const irept &i, const irep_idt &k)
{
  // Mirror irept::is_comment: '#'-prefixed names live in the comments map.
  const std::string &ks = k.as_string();
  const irept::named_subt &m =
    (!ks.empty() && ks[0] == '#') ? i.get_comments() : i.get_named_sub();
  return m.find(k) != m.end();
}

// CBMC stores constant values (integer and floating-point alike) as hex;
// ESBMC wants a binary string whose length matches the constant's own type
// width. The Rust reference (`format!("{:032b}", ...)`) hardcoded 32, which
// truncated the representation of constants in wider (e.g. 64-bit) types: a
// 64-bit value was emitted as a <=33-char string and silently interpreted at
// 32 bits, so e.g. -5000000000LL verified as its low 32 bits (roadmap §4.3/§7).
// Pad to `width` bits instead.
std::string hex_to_bin(const std::string &hex, std::size_t width)
{
  std::string bits;
  if (hex.size() > 16)
  {
    // > 64 bits (e.g. a 128-bit float128 / long double constant): the value no
    // longer round-trips through uint64_t, so expand each hex digit to its four
    // bits directly. A non-hex character means the string is not hex after all
    // (already binary, or something unexpected) -- leave it unchanged.
    bits.reserve(hex.size() * 4);
    for (char ch : hex)
    {
      int v;
      if (ch >= '0' && ch <= '9')
        v = ch - '0';
      else if (ch >= 'a' && ch <= 'f')
        v = ch - 'a' + 10;
      else if (ch >= 'A' && ch <= 'F')
        v = ch - 'A' + 10;
      else
        return hex;
      for (int b = 3; b >= 0; --b)
        bits.push_back(static_cast<char>('0' + ((v >> b) & 1)));
    }
  }
  else
  {
    unsigned long long n = std::stoull(hex, nullptr, 16);
    if (n == 0)
      bits = "0";
    else
      while (n != 0)
      {
        bits.push_back(static_cast<char>('0' + (n & 1)));
        n >>= 1;
      }
    std::reverse(bits.begin(), bits.end());
  }
  if (bits.size() < width)
    bits = std::string(width - bits.size(), '0') + bits;
  return bits;
}

bool irep_contains(const irept &i, const irep_idt &id)
{
  if (i.id() == id)
    return true;
  for (const auto &s : i.get_sub())
    if (irep_contains(s, id))
      return true;
  for (const auto &p : i.get_named_sub())
    if (irep_contains(p.second, id))
      return true;
  for (const auto &p : i.get_comments())
    if (irep_contains(p.second, id))
      return true;
  return false;
}

// Width of a bitvector type irep; falls back to 32 if absent/malformed.
std::size_t bv_width(const irept &bv_type)
{
  const std::string ws = bv_type.find("width").id_string();
  if (!ws.empty() && ws.find_first_not_of("0123456789") == std::string::npos)
    return static_cast<std::size_t>(std::stoul(ws));
  return 32;
}

// Lowercase hex rendering of a small integer (no "0x").
std::string int_to_hex(unsigned long long value)
{
  if (value == 0)
    return "0";
  static const char *digits = "0123456789abcdef";
  std::string s;
  while (value != 0)
  {
    s.push_back(digits[value & 0xF]);
    value >>= 4;
  }
  std::reverse(s.begin(), s.end());
  return s;
}

// A "constant" irep of the given bitvector type holding `value`. The value is
// emitted as a hex string so fix_expression's own constant branch converts it
// to a binary string of the type's exact width on the later recursion (the same
// path CBMC's own constants take -- see hex_to_bin), rather than duplicating
// that width logic here.
irept mk_bv_const(const irept &bv_type, unsigned long long value)
{
  irept c("constant");
  c.add("type") = bv_type;
  c.add("value") = mk(int_to_hex(value));
  return c;
}

// A unary/binary expression node with raw operands in get_sub(); the later
// wrap step in fix_expression moves them into "operands" and the recursion
// normalises them, exactly as for CBMC's own nodes.
irept mk_unary(const irep_idt &id, const irept &type, const irept &op)
{
  irept e(id);
  e.add("type") = type;
  e.get_sub().push_back(op);
  return e;
}

irept mk_binary(
  const irep_idt &id,
  const irept &type,
  const irept &lhs,
  const irept &rhs)
{
  irept e(id);
  e.add("type") = type;
  e.get_sub().push_back(lhs);
  e.get_sub().push_back(rhs);
  return e;
}

// Element type of a complex type irep, tolerant of both shapes: CBMC's raw
// form keeps it as the sole positional sub; the normalised form (fix_type /
// fix_expression's own type case) as the "subtype" named-sub.
irept complex_elem_type(const irept &ctype)
{
  if (has_sub(ctype, "subtype"))
    return ctype.find("subtype");
  if (!ctype.get_sub().empty())
    return ctype.get_sub()[0];
  return irept();
}

// A member access on a complex value's "real"/"imag" component; "member" is
// in fix_expression's operand-wrap set.
irept complex_member(const irept &op, const char *name, const irept &elem)
{
  irept m("member");
  m.add("type") = elem;
  m.set("component_name", name);
  m.get_sub().push_back(op);
  return m;
}

void fix_expression(irept &irep)
{
  if (irep.id() == "is_dynamic_object")
  {
    // __CPROVER_DYNAMIC_OBJECT(p) -- true iff p points into a heap-allocated
    // object -- lowers to an is_dynamic_object expr that migrate_expr has no
    // handler for (it abort()s with "migrate expr failed"). ESBMC tracks the
    // same fact in its symex-managed `c:@__ESBMC_is_dynamic` bool array
    // (indexed by pointer object, set on malloc -- symex_assign.cpp), so the
    // faithful rewrite is `__ESBMC_is_dynamic[pointer_object(p)]`. That needs
    // the array force-linked even when the binary never calls malloc (it is
    // otherwise absent) and its `__ESBMC_inf_size` shape preserved, so it is
    // left as future work. Only an explicit __CPROVER_DYNAMIC_OBJECT reaches
    // here -- CBMC's free()/dynamic checks live in library bodies it re-links
    // at analysis time, not in the serialised binary -- so decline cleanly (a
    // throw the create_goto_program handler turns into a graceful error exit,
    // roadmap §4.7) rather than abort()-ing.
    throw std::string(
      "CBMC adapter: __CPROVER_DYNAMIC_OBJECT (is_dynamic_object) is not yet "
      "supported on the --binary path");
  }

  if (irep.id() == "count_leading_zeros" || irep.id() == "count_trailing_zeros")
  {
    // CBMC lowers __builtin_clz/__builtin_ctz to these expression ids, which
    // migrate_expr has no handler for (it aborts with "migrate expr failed").
    // ESBMC's native path resolves __builtin_clz at symex time
    // (run_builtin.cpp) via a popcount-based bit-count formula; there is no
    // clz/ctz irep2 node. Reproduce that formula here in terms of ids
    // migrate_expr already lowers (bitor, lshr, bitand, bitnot, "-", popcount),
    // so no new node is needed. Scoped to the CBMC --binary path, so it never
    // perturbs native handling (which never emits count_{leading,trailing}_zeros
    // as an expression). clz(0)/ctz(0) is UB; CBMC emits its own #bounds_check
    // guard for the zero argument, which is matched independently.
    const bool leading = irep.id() == "count_leading_zeros";
    const irept operand = irep.get_sub().empty() ? irept() : irep.get_sub()[0];
    const irept optype = operand.find("type"); // operand's bitvector type
    const irept restype = irep.find("type");   // int result type (signedbv/32)
    const std::size_t width = bv_width(optype);

    if (leading)
    {
      // clz(x) = width - popcount(x with every bit below its most-significant
      // set bit smeared down), matching run_builtin.cpp exactly.
      irept smeared = operand;
      for (std::size_t shift = 1; shift < width; shift <<= 1)
        smeared = mk_binary(
          "bitor",
          optype,
          smeared,
          mk_binary("lshr", optype, smeared, mk_bv_const(optype, shift)));

      irept popcount("popcount");
      popcount.add("type") = restype; // migrate forces int32; kept well-formed
      popcount.get_sub().push_back(smeared);

      irep.id("-");
      irep.get_sub().clear();
      irep.get_sub().push_back(mk_bv_const(restype, width));
      irep.get_sub().push_back(popcount);
    }
    else
    {
      // ctz(x) = popcount(~x & (x - 1)): the run of trailing-zero bit positions
      // (x-1 borrows through them, ~x keeps exactly those bits).
      irept arg = mk_binary(
        "bitand",
        optype,
        mk_unary("bitnot", optype, operand),
        mk_binary("-", optype, operand, mk_bv_const(optype, 1)));

      irep.id("popcount");
      irep.get_sub().clear();
      irep.get_sub().push_back(arg);
    }
  }

  if (irep.id() == "rol" || irep.id() == "ror")
  {
    // CBMC lowers __builtin_rotateleft{8,16,32,64}/__builtin_rotateright... to
    // rol/ror expression ids, which migrate_expr has no handler for (aborts with
    // "migrate expr failed"); ESBMC has no rotate irep2 node either. As with
    // clz/ctz above, reproduce the rotate from ids migrate_expr already lowers
    // (shl, lshr, bitor, bitand, "-"), so no new node is needed:
    //   rol(x, n) = (x << d) | (x >> (W - d)),  ror(x, n) = (x >> d) | (x << (W - d))
    // where d = n mod W. CBMC takes the distance mod the width (rol(x, W) == x,
    // rol(x, W + k) == rol(x, k)); W is always a power of two, so `& (W - 1)` is
    // the modulus. The complement (W - d) is also masked with (W - 1) so that
    // d == 0 yields a 0 shift rather than a full-width shift: rol(x, 0) then
    // reduces to (x << 0) | (x >> 0) == x. Scoped to the CBMC --binary path.
    const bool left = irep.id() == "rol";
    const irept x = irep.get_sub().empty() ? irept() : irep.get_sub()[0];
    const irept n = irep.get_sub().size() > 1 ? irep.get_sub()[1] : irept();
    const irept optype = x.find("type"); // value's bitvector type
    const std::size_t width = bv_width(optype);

    // d = n & (W - 1); co = (W - d) & (W - 1)  (both in the value's width)
    const irept d =
      mk_binary("bitand", optype, n, mk_bv_const(optype, width - 1));
    const irept co = mk_binary(
      "bitand",
      optype,
      mk_binary("-", optype, mk_bv_const(optype, width), d),
      mk_bv_const(optype, width - 1));

    irept lo = mk_binary(left ? "shl" : "lshr", optype, x, d);
    irept hi = mk_binary(left ? "lshr" : "shl", optype, x, co);

    irep.id("bitor");
    irep.get_sub().clear();
    irep.get_sub().push_back(lo);
    irep.get_sub().push_back(hi);
  }

  if (irep.id() == "find_first_set")
  {
    // CBMC lowers __builtin_ffs/ffsl/ffsll to a find_first_set irep, which
    // migrate_expr has no handler for (it aborts with "migrate expr failed").
    // find_first_set(x) is the 1-based index of the least-significant set bit,
    // or 0 when x is zero -- exactly __builtin_ffs. ESBMC's native path does not
    // model ffs at all, so, as with clz/ctz above, reproduce it here in terms of
    // ids migrate_expr already lowers rather than adding an irep2 node:
    //   ffs(x) = (x == 0) ? 0 : popcount(~x & (x - 1)) + 1
    // The popcount term is exactly ctz(x) (see the count_trailing_zeros rewrite
    // above); the x == 0 guard is load-bearing because ~0 & (0 - 1) is all-ones,
    // whose popcount is the width, not 0. Scoped to the CBMC --binary path, so it
    // never perturbs native handling (which never emits find_first_set).
    const irept operand = irep.get_sub().empty() ? irept() : irep.get_sub()[0];
    const irept optype = operand.find("type"); // operand's bitvector type
    const irept restype = irep.find("type");   // int result type (signedbv/32)

    // ctz(x) = popcount(~x & (x - 1)), reusing the trailing-zeros formula.
    irept ctz_arg = mk_binary(
      "bitand",
      optype,
      mk_unary("bitnot", optype, operand),
      mk_binary("-", optype, operand, mk_bv_const(optype, 1)));
    irept popcount("popcount");
    popcount.add("type") = restype; // migrate forces int32; kept well-formed
    popcount.get_sub().push_back(ctz_arg);

    // ctz(x) + 1, then guard the zero input: (x == 0) ? 0 : ctz(x) + 1.
    irept ctz_plus1 =
      mk_binary("+", restype, popcount, mk_bv_const(restype, 1));
    irept bool_ty("bool");
    irept is_zero = mk_binary("=", bool_ty, operand, mk_bv_const(optype, 0));

    irep.id("if");
    irep.get_sub().clear();
    irep.get_sub().push_back(is_zero);
    irep.get_sub().push_back(mk_bv_const(restype, 0));
    irep.get_sub().push_back(ctz_plus1);
  }

  if (irep.id() == "bitreverse")
  {
    // CBMC lowers __builtin_bitreverse{8,16,32,64} to a bitreverse irep (reverse
    // the bit order: bit i <-> bit W-1-i), which migrate_expr has no handler for
    // (aborts with "migrate expr failed"); ESBMC has no bitreverse irep2 node.
    // As with clz/ctz above, reproduce it from ids migrate_expr already lowers
    // (bitand, shl, lshr, bitor) via the standard SWAR reversal: swap adjacent
    // bits, then 2-bit groups, then 4-bit, ... doubling the group size each step
    //   acc = ((acc & mask_k) << k) | ((acc >> k) & mask_k)
    // where mask_k selects the low k bits of every 2k-bit block. Scoped to the
    // CBMC --binary path, so it never perturbs native handling.
    const irept operand = irep.get_sub().empty() ? irept() : irep.get_sub()[0];
    const irept optype = operand.find("type"); // value's bitvector type
    const std::size_t width = bv_width(optype);

    irept acc = operand;
    for (std::size_t k = 1; k < width; k <<= 1)
    {
      // mask_k: low k bits set in each 2k-bit block, across the full width.
      const unsigned long long block = (k >= 64) ? ~0ULL : ((1ULL << k) - 1);
      unsigned long long m = 0;
      for (std::size_t pos = 0; pos < width; pos += 2 * k)
        m |= block << pos;
      const irept mask = mk_bv_const(optype, m);

      // acc = ((acc & mask) << k) | ((acc >> k) & mask)
      irept lo = mk_binary(
        "shl",
        optype,
        mk_binary("bitand", optype, acc, mask),
        mk_bv_const(optype, k));
      irept hi = mk_binary(
        "bitand",
        optype,
        mk_binary("lshr", optype, acc, mk_bv_const(optype, k)),
        mask);
      acc = mk_binary("bitor", optype, lo, hi);
    }

    irep = acc;
  }

  if (irep.id() == "sign" && irep.find("type").id() == "bool")
  {
    // CBMC's sign-bit predicate "sign" is bool-typed and used directly in
    // boolean contexts (e.g. the ternary condition in glibc's isinf). ESBMC's
    // equivalent irep, "signbit", is structurally fixed to int32 (1 iff the
    // sign bit is set), so it can't stand in for a bool. Rewrite sign(x) to
    // the boolean signbit(x) != 0 -- exactly the form ESBMC's own C frontend
    // produces for a sign-bit test in boolean context (clang_c_adjust_expr.cpp
    // __builtin_isinf_sign) -- so it type-checks where CBMC uses it. Scoped to
    // this CBMC path, so it never perturbs ESBMC's native signbit handling.
    irept operand;
    if (!irep.get_sub().empty())
      operand = irep.get_sub()[0];

    irept int_ty("signedbv");
    int_ty.set("width", "32");

    irept signbit("signbit");
    signbit.add("type") = int_ty;
    signbit.get_sub().push_back(operand);

    irept zero("constant");
    zero.add("type") = int_ty;
    zero.add("value") = mk(std::string(32, '0'));

    // Rebuild the node as notequal(signbit(x), 0); its bool type is retained.
    // The operand-wrap step below moves [signbit, zero] into "operands" and
    // the recursion normalises them (signbit's operand, the zero constant).
    irep.id("notequal");
    irep.get_sub().clear();
    irep.get_sub().push_back(signbit);
    irep.get_sub().push_back(zero);
  }

  if (
    (irep.id() == "forall" || irep.id() == "exists") &&
    irep.get_sub().size() == 2 && irep.get_sub()[0].id() == "tuple")
  {
    // CBMC binds a quantifier's variable(s) inside a "tuple" node in the first
    // operand; ESBMC's forall2t/exists2t (and the solver, smt_solver.cpp) expect
    // side_1 to be the bound symbol itself. Unwrap a single-symbol tuple to the
    // bare symbol. A tuple with more than one bound variable is left untouched
    // (forall2t binds exactly one symbol) so it aborts cleanly rather than
    // silently dropping the extra binders -- a soundness hazard.
    const irept &tuple = irep.get_sub()[0];
    if (tuple.get_sub().size() == 1)
    {
      // Copy the bound symbol out before overwriting the slot that holds the
      // tuple (the source lives inside it), mirroring fix_builtin_call's
      // copy-before-mutate discipline.
      const irept bound = tuple.get_sub()[0];
      irep.get_sub()[0] = bound;
    }
  }

  if (irep.id() == "complex" && has_sub(irep, "type"))
    // The complex *constructor* expression (real, imag operands) -- CBMC also
    // uses the id "complex" for the _Complex *type* itself, the same
    // type/expression ambiguity as "array" above; an expression always
    // carries a "type" named-sub, a type never does (types are normalised by
    // fix_type, which runs over the whole function irep afterwards). ESBMC's
    // own C frontend builds a complex value as a "struct" expr over the
    // complex type's (real, imag) components (clang_c_convert.cpp), and
    // constant_struct2t explicitly admits a complex-typed aggregate. "struct"
    // is in the operand-wrap set below.
    irep.id("struct");

  if (
    irep.id() == "typecast" && irep.find("type").id() == "complex" &&
    !irep.get_sub().empty())
  {
    // Casts *to* complex. There is no complex typecast in ESBMC's SMT layer;
    // its own C frontend lowers a real -> complex cast to a {value, 0}
    // aggregate and a complex -> complex cast to a component-wise cast
    // (clang_c_convert.cpp, CK_FloatingRealToComplex / CK_FloatingComplexCast).
    const irept elem = complex_elem_type(irep.find("type"));
    const irept op = irep.get_sub()[0];
    irept re, im;
    if (op.find("type").id() == "complex")
    {
      re = complex_member(op, "real", complex_elem_type(op.find("type")));
      im = complex_member(op, "imag", complex_elem_type(op.find("type")));
    }
    else
    {
      re = op;
      im = mk_bv_const(elem, 0);
    }
    if (!(re.find("type") == elem))
      re = mk_unary("typecast", elem, re);
    if (!(im.find("type") == elem))
      im = mk_unary("typecast", elem, im);
    irep.id("struct");
    irep.get_sub().clear();
    irep.get_sub().push_back(re);
    irep.get_sub().push_back(im);
  }
  else if (
    irep.id() == "typecast" && !irep.get_sub().empty() &&
    irep.get_sub()[0].find("type").id() == "complex")
  {
    // Casts *from* complex to a real type: C99 6.3.1.7 discards the imaginary
    // part, and ESBMC's own C frontend lowers this to a member access on the
    // real component (clang_c_convert.cpp, CK_FloatingComplexToReal). Without
    // this the raw typecast reaches the solver and aborts ("Typecast for
    // unexpected type"). The complex -> _Bool form does not arise: goto-cc
    // 6.8.0 itself crashes typechecking it, so no binary can contain one.
    const irept op = irep.get_sub()[0];
    const irept elem = complex_elem_type(op.find("type"));
    irept re = complex_member(op, "real", elem);
    if (irep.find("type") == elem)
      irep = re;
    else
      irep.get_sub()[0] = re;
  }

  if (
    (irep.id() == "+" || irep.id() == "-" || irep.id() == "*" ||
     irep.id() == "/") &&
    irep.find("type").id() == "complex" && irep.get_sub().size() >= 2)
  {
    // Complex arithmetic. There is no complex add/sub/mul/div in ESBMC's SMT
    // layer; its own C frontend lowers these component-wise over the (real,
    // imag) members at adjust time (clang_c_adjust_expr.cpp::
    // adjust_expr_binary_arithmetic). Mirror that lowering here.
    //
    // CBMC's +/* nodes can be n-ary (its simplifier may flatten a+b+c);
    // pre-fold to nested binaries, left-associatively, so the component-wise
    // lowering below only ever sees a pair. The nested node is complex-typed
    // and lands inside the member operands, where the later recursion lowers
    // it through this same case.
    while (irep.get_sub().size() > 2)
    {
      irept inner(irep.id());
      inner.add("type") = irep.find("type");
      inner.get_sub().push_back(irep.get_sub()[0]);
      inner.get_sub().push_back(irep.get_sub()[1]);
      irep.get_sub().erase(irep.get_sub().begin());
      irep.get_sub()[0] = inner;
    }
    const irept ctype = irep.find("type");
    const irept elem = complex_elem_type(ctype);

    // Promote a non-complex operand to {val, 0}, as the native frontend does.
    auto promote = [&](irept e) {
      if (e.find("type").id() == "complex")
        return e;
      irept s("struct");
      s.add("type") = ctype;
      s.get_sub().push_back(e);
      s.get_sub().push_back(mk_bv_const(elem, 0));
      return s;
    };
    const irept a = promote(irep.get_sub()[0]);
    const irept b = promote(irep.get_sub()[1]);

    // Component ops on a floatbv element must be the ieee_ forms, exactly as
    // the float-arithmetic promotion below rewrites scalar float ops.
    const bool fp = elem.id() == "floatbv";
    auto mk_op = [&](const char *id, const irept &lhs, const irept &rhs) {
      static const std::unordered_map<std::string, std::string> ieee = {
        {"+", "ieee_add"},
        {"-", "ieee_sub"},
        {"*", "ieee_mul"},
        {"/", "ieee_div"}};
      return mk_binary(fp ? ieee.at(id) : irep_idt(id), elem, lhs, rhs);
    };

    const irept ar = complex_member(a, "real", elem);
    const irept ai = complex_member(a, "imag", elem);
    const irept br = complex_member(b, "real", elem);
    const irept bi = complex_member(b, "imag", elem);
    irept new_real, new_imag;
    const std::string op = irep.id_string();
    if (op == "+" || op == "-")
    {
      new_real = mk_op(op.c_str(), ar, br);
      new_imag = mk_op(op.c_str(), ai, bi);
    }
    else if (op == "*")
    {
      new_real = mk_op("-", mk_op("*", ar, br), mk_op("*", ai, bi));
      new_imag = mk_op("+", mk_op("*", ar, bi), mk_op("*", ai, br));
    }
    else
    {
      const irept denom = mk_op("+", mk_op("*", br, br), mk_op("*", bi, bi));
      new_real =
        mk_op("/", mk_op("+", mk_op("*", ar, br), mk_op("*", ai, bi)), denom);
      new_imag =
        mk_op("/", mk_op("-", mk_op("*", ai, br), mk_op("*", ar, bi)), denom);
    }

    irep.id("struct");
    irep.get_sub().clear();
    irep.get_sub().push_back(new_real);
    irep.get_sub().push_back(new_imag);
  }

  if (
    irep.id() == "unary-" && irep.find("type").id() == "complex" &&
    !irep.get_sub().empty())
  {
    // Complex negation, component-wise like the binary arithmetic above.
    // (ESBMC's native frontend does not lower unary complex ops at all --
    // -z segfaults on the native path -- so the CBMC path gains coverage
    // native still lacks, as with ctz. GNU ~z conjugation is deliberately
    // NOT handled: CBMC 6.8.0 itself mis-models it, so there is no parity
    // target to match.)
    const irept elem = complex_elem_type(irep.find("type"));
    const irept op = irep.get_sub()[0];
    irep.id("struct");
    irep.get_sub().clear();
    irep.get_sub().push_back(
      mk_unary("unary-", elem, complex_member(op, "real", elem)));
    irep.get_sub().push_back(
      mk_unary("unary-", elem, complex_member(op, "imag", elem)));
  }

  if (irep.id() == "complex_real" || irep.id() == "complex_imag")
  {
    // __real__ / __imag__ accessors. ESBMC's own C frontend lowers these to a
    // member access on the complex value's (real, imag) components
    // (clang_c_convert.cpp, CK_FloatingComplexToReal / UO_Real); migrate_expr
    // has no complex_real/complex_imag handler. "member" is in the operand-wrap
    // set below.
    irep.set("component_name", irep.id() == "complex_real" ? "real" : "imag");
    irep.id("member");
  }

  if (irep.id() == "side_effect")
    irep.id("sideeffect");
  else if (irep.id() == "string_constant")
    irep.id("string-constant");
  else if (irep.id() == "ieee_float_equal")
    irep.id("=");
  else if (irep.id() == "ieee_float_notequal")
    // CBMC's IEEE-754 float inequality (NaN != NaN is true) has no migrate_expr
    // handler, so it aborts with "migrate expr failed". ESBMC's own C frontend
    // lowers a float != to a plain "notequal" whose floatbv SMT encoding already
    // implements IEEE semantics (NaN-aware), so rewrite to that -- the exact
    // counterpart of the "ieee_float_equal" -> "=" rewrite above. "notequal" is
    // in the operand-wrap set below, so its operands reach migrate_expr.
    irep.id("notequal");
  else if (
    (irep.id() == "+" || irep.id() == "-" || irep.id() == "*" ||
     irep.id() == "/") &&
    irep.find("type").id() == "floatbv")
  {
    // CBMC emits plain +/-/*// for float arithmetic, type-blind; ESBMC's own
    // C frontend promotes these to their ieee_ counterparts whenever the
    // type is floatbv (clang_c_adjust_expr.cpp::adjust_float_arith), and
    // downstream checks rely on that distinction (e.g. goto_check.cpp skips
    // the division-by-zero check for ieee_div, since it's defined IEEE-754
    // behaviour, not UB). Mirror that promotion here.
    if (irep.id() == "+")
      irep.id("ieee_add");
    else if (irep.id() == "-")
      irep.id("ieee_sub");
    else if (irep.id() == "*")
      irep.id("ieee_mul");
    else
      irep.id("ieee_div");
  }

  if (irep.id() == "constant")
  {
    const std::string type_id = irep.find("type").id_string();
    if (type_id != "pointer" && type_id != "bool" && has_sub(irep, "value"))
    {
      const std::string val = irep.find("value").id_string();
      const std::size_t width = bv_width(irep.find("type"));
      // CBMC stores the value as hex; ESBMC wants a binary string of the type's
      // own bit width. A value already exactly `width` chars long is an existing
      // binary string (e.g. one this pass produced earlier) and is left as-is;
      // anything else is hex and gets converted (see hex_to_bin). Keying on the
      // type width, not a hardcoded 32, is what makes 128-bit float128 / long
      // double work: its value is 32 hex chars, which a `!= 32` guard mistook
      // for an already-binary 32-bit value and left as raw hex, so migrate
      // misdecoded it (1.5L read as ~0 -- a false verdict, not a crash).
      if (val.size() != width)
        irep.add("value") = mk(hex_to_bin(val, width));
    }
  }

  static const std::unordered_set<std::string> expressions = {
    "if",
    "member",
    "typecast",
    "notequal",
    "and",
    "or",
    // Boolean implication (a ==> b); migrate_expr lowers "=>" to implies2t via a
    // wrapped operand pair. Common in quantifier bodies (__CPROVER_forall guards)
    // but valid in any boolean context.
    "=>",
    "mod",
    "not",
    "*",
    "/",
    "+",
    "-",
    "ieee_add",
    "ieee_sub",
    "ieee_mul",
    "ieee_div",
    "=",
    "<",
    "<=",
    ">",
    ">=",
    "unary-",
    "overflow_result-+",
    "overflow_result--",
    "overflow_result-*",
    "overflow_result-shr",
    // Overflow *predicates* (distinct from overflow_result, which returns the
    // value+flag pair): CBMC's `overflow-<op>` bool node, which __builtin_add/
    // sub/mul_overflow_p lower to. migrate_expr handles the whole family via
    // convert_operand_pair, so each just needs its operands wrapped (same
    // requirement as popcount/isnan); +/-/* are exercised by the builtins, the
    // rest share the identical mechanism.
    "overflow-+",
    "overflow--",
    "overflow-*",
    "overflow-/",
    "overflow-mod",
    "overflow-shl",
    "overflow-unary-",
    "lshr",
    "ashr",
    "shl",
    "address_of",
    "index",
    "byte_extract_little_endian",
    "pointer_object",
    "pointer_offset",
    "array_of",
    "sideeffect",
    "dereference",
    "object_size",
    "bitand",
    "bitor",
    "bitxor",
    "bitnot",
    "struct",
    "union",
    "return",
    "r_ok",
    "w_ok",
    "rw_ok",
    "isnan",
    "isinf",
    "isnormal",
    "isfinite",
    "nearbyint",
    "signbit",
    "ieee_sqrt",
    "ieee_fma",
    "abs",
    // Unary bit-builtins: migrate_expr already handles popcount/bswap via op0(),
    // but without wrapping CBMC's raw operands into "operands" here, op0() reads
    // an empty list (same failure shape as isnan/pointer_offset). __builtin_bswap
    // / __builtin_popcount lower to these ids in CBMC's goto.
    "popcount",
    "bswap",
    // Quantifier predicates: __CPROVER_forall/__CPROVER_exists lower to these
    // ids, which migrate_expr handles via op0()/op1() (bound symbol, predicate).
    // Without wrapping CBMC's raw operands into "operands" here, op0() reads an
    // empty operand list and segfaults (same failure shape as isnan/popcount).
    "forall",
    "exists"};

  const std::string cur = irep.id_string();

  // In CBMC both an expression and a type can be named "array"; "array" is an
  // umbrella expression that can also mean index.
  const bool array_has_operand = cur == "array" && has_sub(irep, "type") &&
                                 irep.find("type").id() == "array" &&
                                 !irep.get_sub().empty();

  const bool is_function_call = cur == "arguments" && !irep.get_sub().empty();

  if (expressions.count(cur) != 0 || array_has_operand || is_function_call)
  {
    irept operands;
    operands.get_sub() = irep.get_sub();
    irep.get_sub().clear();
    irep.add("operands") = operands;
  }

  for (auto &s : irep.get_sub())
    fix_expression(s);

  for (auto &p : irep.get_named_sub())
  {
    if (p.first == "components")
      for (auto &c : p.second.get_sub())
        c.id("component");
    fix_expression(p.second);
  }
}

// CBMC's anonymous-aggregate naming convention (roadmap §4.3).
bool is_anon_tag(const std::string &ident)
{
  return ident.size() >= 11 && ident.compare(0, 10, "tag-#anon#") == 0;
}

// A struct_tag/union_tag whose definition is not yet in the cache. CBMC emits
// an anonymous aggregate's type symbol *after* the struct that contains it (the
// aggregate tag encodes the container's layout), so the first fix pass reaches
// the container's anonymous member before that symbol is cached. Leave the tag
// unresolved -- exactly as fix_type already does for a not-yet-seen *named* tag
// -- so the re-check pass in adapt_cbmc_to_esbmc, which runs with the full
// cache, resolves it. Resolving from CBMC's own serialised type symbol (rather
// than parsing the tag-name grammar) guarantees the definition is byte-identical
// to the one the reader builds for the same member in an instruction, which
// with2t::assert_type_compat_for_with compares by value. A tag that is still
// unresolved after the re-check pass trips that pass's own
// "should have been resolved" guard.
void expand_anon_struct(const irept &)
{
}

// Decline CPROVER's symbolic `string` type, which reaches the adapter from
// every JBMC binary: the Java frontend types java.lang.Object's
// @class_identifier with it, and every Java class embeds java.lang.Object, so
// it surfaces during type migration before any instruction is examined. ESBMC
// has no string type -- migrate_type's fall-through logs a bare type dump,
// which is why the failure reads as the unhelpful "ERROR: string".
// Representing it means first deciding how class tags are modelled
// (docs/jbmc-goto-binary-poc-plan.md §2.3.1 records the evidence), so decline
// rather than guess at a mapping.
//
// This must only fire in a *type* position. fix_type walks whole symbols and
// whole function bodies, so it also visits identifier nodes, whose id() is the
// identifier text itself -- and `string` is an ordinary C identifier. A bare
// string_typet serialises as {"id": "string"} with no subs, structurally
// indistinguishable from such a node, so position is the only usable signal.
void reject_string_type(const irept &candidate)
{
  if (candidate.id() == "string")
    throw std::string(
      "CBMC adapter: the 'string' type (Java's @class_identifier) is not yet "
      "supported on the --binary path");
}

// The named-sub keys under which a type -- and never an identifier -- appears.
// Everything else (name, identifier, base_name, ...) holds the identifier text
// as its id(), so descending one of those leaves type position.
bool is_type_edge(const irep_idt &key)
{
  return key == "type" || key == "subtype" || key == "return_type";
}

// `expanding` holds the tag identifiers whose definitions are currently being
// inlined on the recursion stack, so a recursive aggregate (e.g. a Rust struct
// holding a pointer to itself) can be detected and broken -- see the tag branch
// below. The two-argument overload seeds it for external callers.
// `in_type_position` records whether `self` was reached through a type edge.
// The rewrites below key on self.id(), which for an identifier node is the
// identifier text -- so without it, a C symbol named `c_bool`, `c_enum_tag` or
// `string` is mistaken for the type of the same name. That is not hypothetical:
// a function named c_bool had its name rewritten to `signedbv`, and the
// resulting binary verified to FAILED on a program that should pass.
void fix_type(
  irept &self,
  const std::unordered_map<std::string, irept> &cache,
  std::unordered_set<std::string> &expanding,
  bool in_type_position)
{
  if (in_type_position)
    reject_string_type(self);

  if (in_type_position && self.id() == "c_bool")
  {
    self.id("signedbv");
    return;
  }

  if (in_type_position && self.id() == "c_enum_tag")
  {
    // CBMC references an enum type via a c_enum_tag node -- the tag counterpart
    // of c_enum, mirroring struct_tag/union_tag. migrate_type maps c_enum/
    // incomplete_c_enum to a signed int (C99 6.7.2.2.3) but has no case for the
    // tag, so any enum-typed object aborts with "ERROR: c_enum_tag". An enum is
    // consistently int-typed and migrate discards the underlying width anyway,
    // so rather than resolve the tag through the cache (which only holds struct/
    // union definitions) rewrite it to a bare c_enum and let migrate yield the
    // same int type.
    self = mk("c_enum");
    return;
  }

  if (self.id() == "c_bit_field")
  {
    // CBMC types a bitfield member as c_bit_field{width: N; sub[0]: <underlying
    // integer bv of width W>}. ESBMC has no c_bit_field type -- migrate_type
    // aborts on it ("ERROR: c_bit_field") -- and instead represents a bitfield
    // exactly as its native C frontend does (clang_c_convert.cpp::
    // get_bitfield_type): the underlying bv kind narrowed to the bitfield width
    // N, tagged #bitfield, carrying the full underlying type as its subtype.
    // migrate_type then reads width N and yields an N-bit bv (get_uint_type(N)
    // for the bool case, signedbv/unsignedbv of N bits otherwise).
    if (self.get_sub().empty())
    {
      // CBMC always emits the underlying integer type as sub[0]; a c_bit_field
      // without it is malformed. Fail loud (like expand_anon_struct) rather
      // than emit an id-less type migrate_type would choke on obscurely.
      log_error("CBMC adapter: c_bit_field without an underlying type");
      abort();
    }
    irept underlying = self.get_sub()[0];
    // A _Bool bitfield must stay boolean: ESBMC's migrate reads a #bitfield
    // bool as an *unsigned* N-bit value (get_uint_type), whereas fix_type maps
    // CBMC's c_bool to signedbv -- a 1-bit signedbv would read value 1 as -1.
    // Detect the bool underlying before that rewrite and keep the result "bool".
    const bool bool_underlying =
      underlying.id() == "bool" || underlying.id() == "c_bool";
    fix_type(underlying, cache, expanding, true);
    const irep_idt bf_width = self.find("width").id();
    self.id(bool_underlying ? irep_idt("bool") : underlying.id());
    self.get_sub().clear();
    self.set("width", bf_width);
    self.set("#bitfield", true);
    self.add("subtype") = underlying;
    return;
  }

  if (self.id() == "code" && has_sub(self, "parameters"))
  {
    irept arguments(irep_idt("arguments"));
    arguments.get_sub() = self.find("parameters").get_sub();
    self.add("arguments") = arguments;
  }

  if (has_sub(self, "components"))
  {
    irept &components = self.add("components");
    for (auto &v : components.get_sub())
      v.id("component"); // fix_struct
  }

  if (self.id() == "pointer")
  {
    // A struct/union tag under a pointer becomes a "symbol" back-reference
    // instead of being inlined -- the shape ESBMC's native frontends produce.
    // C recursion necessarily passes through a pointer (a struct cannot
    // contain itself by value), so this keeps every spelling of a recursive
    // type identical; inlining here left the symbol table with a
    // once-unrolled definition while instruction types were depth-1, and
    // comparisons between the two spellings either recursed without bound
    // (stack overflow in type2t::cmp) or reported a false "incompatible
    // base type". Anonymous tags cannot be recursive and ESBMC-side
    // resolution of their identifiers is unproven, so they keep the inline
    // path. Both pointee spellings must be handled: the raw positional sub,
    // and the already-moved "subtype" named-sub when a later pass re-fixes
    // with a fuller cache (mutually-recursive definitions resolve only then).
    irept *pointee = nullptr;
    if (has_sub(self, "subtype"))
      pointee = &self.add("subtype");
    else if (!self.get_sub().empty())
      pointee = &self.get_sub()[0];
    if (
      pointee &&
      (pointee->id() == "struct_tag" || pointee->id() == "union_tag") &&
      has_sub(*pointee, "identifier"))
    {
      const std::string ident = pointee->find("identifier").id_string();
      if (!is_anon_tag(ident) && cache.count(ident) != 0)
      {
        irept ref = mk("symbol");
        ref.identifier(ident);
        *pointee = ref;
      }
    }
  }

  if (
    self.id() == "pointer" && !has_sub(self, "subtype") &&
    !self.get_sub().empty())
  {
    for (auto &v : self.get_sub())
      fix_type(v, cache, expanding, true);
    // The pointed-to type is the sole positional sub, exactly like the array
    // case below -- it must be assigned directly, not wrapped in an
    // intermediate group irep. typet::subtype() (util/type.h) is a direct
    // find("subtype"): wrapping here would make it return the wrapper
    // instead of the real type, silently downgrading every such pointer to
    // void* once migrate_type resolves it (irep2.h has no case for an
    // id-less group irep).
    irept magic = self.get_sub()[0];
    self.add("subtype") = magic;
    self.get_sub().clear();
  }

  if (
    self.id() == "array" && !has_sub(self, "subtype") &&
    !self.get_sub().empty())
  {
    irept magic = self.get_sub()[0];
    self.add("subtype") = magic;
    self.get_sub().clear();
    // CBMC can't decide whether array sizes are binary or hex.
    for (auto &p : self.get_named_sub())
      if (p.first == "size" && has_sub(p.second, "value"))
        fix_expression(p.second);
  }

  if (
    self.id() == "complex" && !has_sub(self, "type") &&
    !has_sub(self, "subtype") && !self.get_sub().empty())
  {
    // _Complex element type is the sole positional sub, exactly like pointer/
    // array above; migrate_type reads it via subtype() (a direct
    // find("subtype")). The !has_sub("type") conjunct keeps this from ever
    // eating a complex *constructor* expression (which fix_expression re-ids
    // to "struct" before fix_type runs -- but that ordering is not otherwise
    // enforced, and treating a constructor as a type would silently drop its
    // imaginary operand).
    irept magic = self.get_sub()[0];
    self.add("subtype") = magic;
    self.get_sub().clear();
  }

  // struct_tag and union_tag are the two aggregate references CBMC emits; both
  // resolve out of the same tag cache into their concrete definition.
  const bool is_tag = self.id() == "struct_tag" || self.id() == "union_tag";
  if (!is_tag)
  {
    for (auto &v : self.get_sub())
      fix_type(v, cache, expanding, in_type_position);
    for (auto &p : self.get_named_sub())
      fix_type(p.second, cache, expanding, is_type_edge(p.first));
    for (auto &p : self.get_comments())
      fix_type(p.second, cache, expanding, false);
    return;
  }

  if (!has_sub(self, "identifier"))
    return;

  const std::string ident = self.find("identifier").id_string();
  auto it = cache.find(ident);
  if (it == cache.end())
  {
    expand_anon_struct(self);
    return;
  }

  if (!expanding.insert(ident).second)
  {
    irept ref = mk("symbol");
    ref.identifier(ident);
    self = ref;
    return;
  }

  self = it->second;

  // The resolved aggregate may itself contain tags; redo the cache walk.
  if (irep_contains(self, "struct_tag") || irep_contains(self, "union_tag"))
  {
    for (auto &v : self.get_sub())
      fix_type(v, cache, expanding, in_type_position);
    for (auto &p : self.get_named_sub())
      fix_type(p.second, cache, expanding, is_type_edge(p.first));
    for (auto &p : self.get_comments())
      fix_type(p.second, cache, expanding, false);
  }

  expanding.erase(ident);
}

// External entry point: fresh expansion stack per top-level type.
void fix_type(irept &self, const std::unordered_map<std::string, irept> &cache)
{
  std::unordered_set<std::string> expanding;
  fix_type(self, cache, expanding, false);
}

// Entry point for a type SYMBOL's own definition: seed the expansion stack
// with the symbol's own tag identity so a self-reference (struct node { ...
// struct node *next; }) becomes a "symbol" back-reference immediately --
// the shape ESBMC's native frontends produce. Without the seed, the
// definition pass inlines the tag into itself once, and the symbol table
// ends up holding a once-unrolled definition while every other spelling of
// the type is depth-1: symbol-following type comparisons between the two
// spellings never align, and symex recurses without bound (stack-overflow
// segfault in type2t::cmp) the moment a pointer-typed member is read
// through a dereference (a.next->next on a linked list).
void fix_type_symbol_definition(
  irept &self,
  const std::unordered_map<std::string, irept> &cache,
  const std::string &self_ident)
{
  std::unordered_set<std::string> expanding{self_ident};
  fix_type(self, cache, expanding, false);
}

// Builds the malloc/alloca side_effect_exprt (irep shape) do_mem would have
// built, falling back to element type char -- do_mem's own fallback whenever
// the size argument isn't a recognisable sizeof(T) pattern, which a CBMC
// binary's argument never is (sizeof is constant-folded away by goto-cc
// well before the .goto file exists). A byte-granularity allocation is a
// sound, if less precise, model of any malloc(n)/alloca(n) call regardless of
// the pointer type it's later cast to. `statement` selects "malloc" (dynamic
// object) or "alloca" (automatic, freed on function return) -- migrate_expr
// maps them to sideeffect2t allockind malloc/alloca respectively.
irept build_mem_rhs(
  const irept &lhs,
  const irept::subt &args,
  const char *statement)
{
  if (args.size() != 1)
    return get_nil_irep();

  // get_alloc_size (goto-programs/builtin_functions.cpp) always coerces the
  // allocation size to size_t; mirror that instead of assuming CBMC's raw
  // argument already has the right width.
  irept size_arg(irep_idt("typecast"));
  size_arg.add("type") = static_cast<const irept &>(size_type());
  size_arg.get_sub().push_back(args[0]);
  // fix_expression only ever recurses into get_sub()/get_named_sub(), never
  // comments -- normalise (e.g. constant hex->binary) explicitly, since this
  // copy is about to be embedded in the "#size" comment, which no later pass
  // will otherwise reach.
  fix_expression(size_arg);

  irept sideeffect(irep_idt("sideeffect"));
  sideeffect.add("type") = lhs.find("type");
  sideeffect.add("statement") = mk(statement);
  sideeffect.get_sub().push_back(args[0]);
  sideeffect.add("#size") = size_arg;
  sideeffect.add("#type") = static_cast<const irept &>(char_type());
  return sideeffect;
}

// Builds a unary floating-point exprt (irep shape) with the given id, mirroring
// what clang_c_adjust_expr.cpp builds for a syntactically-recognised libm call
// (sqrt/sqrtf/sqrtl -> ieee_sqrt, nearbyint/nearbyintf/nearbyintl ->
// nearbyint). No explicit rounding_mode operand -- migrate_expr's ieee_sqrt /
// nearbyint handlers default to the standard c:@__ESBMC_rounding_mode symbol
// when one isn't present, same as the ieee_add/sub/mul/div family. That default
// symbol is defined by esbmc_parseoptions.cpp's synthesize_cprover_additions,
// which runs before read_cbmc_goto_object in every *normal* --binary
// invocation, but not under --no-cprover-additions -- currently masked there by
// an unrelated, pre-existing entry-point resolution gap, not a live bug today,
// but not this function's to assume away either.
irept build_unary_fp_rhs(
  const irept &lhs,
  const irept::subt &args,
  const char *id)
{
  if (args.size() != 1)
    return get_nil_irep();

  const irep_idt expr_id(id);
  irept result(expr_id);
  result.add("type") = lhs.find("type");
  result.get_sub().push_back(args[0]);
  return result;
}

// Builds the (ptr == NULL) ? malloc(size) : realloc(ptr) conditional
// goto_convertt::do_realloc produces for realloc(ptr, size). The null guard is
// load-bearing: symex_realloc assumes a live source object, so realloc(NULL, n)
// must route through the malloc side-effect (C says realloc(NULL, n) == malloc
// (n)). The malloc branch reuses build_mem_rhs; the realloc branch is a
// side_effect("realloc", ptr) carrying the byte size in "#size", which
// migrate_expr maps to sideeffect2t allockind realloc -> symex_realloc. The
// "if"/"="/sideeffect ids are all in fix_expression's wrap-set, so each node's
// operands are wrapped by the later recursion in instruction_to_esbmc_irep.
irept build_realloc_rhs(const irept &lhs, const irept::subt &args)
{
  if (args.size() != 2)
    return get_nil_irep();

  const irept ptr = args[0];
  const irept size = args[1];

  irept::subt size_only;
  size_only.push_back(size);
  irept malloc_branch = build_mem_rhs(lhs, size_only, "malloc");
  if (malloc_branch.is_nil())
    return get_nil_irep();

  // realloc branch: side_effect("realloc", ptr), #size = size coerced to size_t
  // (mirrors build_mem_rhs; get_alloc_size always coerces the allocation
  // size). fix_expression only recurses into get_sub()/get_named_sub(), never
  // comments, so normalise the "#size" copy explicitly here.
  irept size_arg(irep_idt("typecast"));
  size_arg.add("type") = static_cast<const irept &>(size_type());
  size_arg.get_sub().push_back(size);
  fix_expression(size_arg);

  irept realloc_branch(irep_idt("sideeffect"));
  realloc_branch.add("type") = lhs.find("type");
  realloc_branch.add("statement") = mk("realloc");
  realloc_branch.get_sub().push_back(ptr);
  realloc_branch.add("#size") = size_arg;

  // is_null = (ptr == NULL); a NULL-valued pointer constant migrates to the
  // null symbol (migrate.cpp).
  irept null_const(irep_idt("constant"));
  null_const.add("type") = ptr.find("type");
  null_const.add("value") = mk("NULL");

  irept is_null(irep_idt("="));
  is_null.add("type") = static_cast<const irept &>(bool_type());
  is_null.get_sub().push_back(ptr);
  is_null.get_sub().push_back(null_const);

  irept if_expr(irep_idt("if"));
  if_expr.add("type") = lhs.find("type");
  if_expr.get_sub().push_back(is_null);
  if_expr.get_sub().push_back(malloc_branch);
  if_expr.get_sub().push_back(realloc_branch);
  return if_expr;
}

// Builds an ieee_fma exprt for fma(a, b, c) = a*b + c (single-rounding fused
// multiply-add). migrate_expr reads op0/op1/op2 and defaults the rounding mode
// like the rest of the ieee_* family (see build_unary_fp_rhs).
irept build_fma_rhs(const irept &lhs, const irept::subt &args)
{
  if (args.size() != 3)
    return get_nil_irep();

  irept result(irep_idt("ieee_fma"));
  result.add("type") = lhs.find("type");
  for (const irept &a : args)
    result.get_sub().push_back(a);
  return result;
}

// __builtin_nan("")/__builtin_nanf("") construct a quiet NaN. CBMC's own
// <builtin-library-__builtin_nan> body returns floatbv_div(0, 0, rounding_mode),
// i.e. 0.0/0.0, so mirror that exactly: ieee_div of two +0.0 constants of the
// result's float type. The NaN-payload string argument is ignored -- it does not
// affect NaN-ness, and ESBMC's own C frontend likewise folds __builtin_nan to a
// constant NaN without dereferencing it. ieee_div is in fix_expression's
// operand-wrap set and defaults its rounding mode like the rest of the ieee_*
// family. Restricted to double/float: CBMC 6.5.0 does not model __builtin_nanl
// as a NaN (its result compares equal to itself, so x != x is FALSE), so nanl is
// deliberately left as a bodyless external -- whose nondet return already yields
// the same FAILED verdict CBMC gives -- to preserve verdict parity.
irept build_nan_rhs(const irept &lhs)
{
  const irept ftype = lhs.find("type");
  irept zero(irep_idt("constant"));
  zero.add("type") = ftype;
  zero.add("value") = mk("0"); // hex 0 -> +0.0 after width-aware conversion
  irept result(irep_idt("ieee_div"));
  result.add("type") = ftype;
  result.get_sub().push_back(zero);
  result.get_sub().push_back(zero);
  return result;
}

// __builtin_huge_val{,f,l} / __builtin_inf{,f,l} construct positive infinity.
// CBMC's <builtin-library-*> bodies return it, but the bodies do not survive the
// reader/adapter (their flattened floatbv nodes have no migrate handler), so
// these reach symex as bodyless externals returning nondet -- and a valid
// `double x = __builtin_huge_val(); assert(x > 1e30)` reports a false FAILED.
// Emit +Inf directly as a floatbv constant: sign 0, exponent all ones, mantissa
// 0. The value is written as the full-width binary bit pattern (fix_expression's
// constant branch leaves an already-width-length string unchanged), which works
// for every width including 128-bit long double -- unlike a 64-bit literal.
// (__builtin_inf -- double, no suffix -- is folded to a constant by CBMC and
// never reaches here; it is matched for uniformity and is harmless.)
irept build_inf_rhs(const irept &lhs)
{
  const irept ftype = lhs.find("type");
  const std::size_t width = bv_width(ftype);
  const std::string fs = ftype.find("f").id_string();
  const std::size_t frac =
    (!fs.empty() && fs.find_first_not_of("0123456789") == std::string::npos)
      ? static_cast<std::size_t>(std::stoul(fs))
      : 0;
  // width = 1 (sign) + exp_bits + frac, so exp_bits = width - 1 - frac.
  const std::size_t exp_bits = (width > frac + 1) ? width - 1 - frac : 0;

  // MSB-first: sign(0), exponent(all ones), fraction(zeros).
  std::string bits(width, '0');
  for (std::size_t i = 1; i <= exp_bits; ++i)
    bits[i] = '1';

  irept c(irep_idt("constant"));
  c.add("type") = ftype;
  c.add("value") = mk(bits);
  return c;
}

// CBMC-sourced FUNCTION_CALL instructions never go through ESBMC's own
// goto_convert, so ESBMC's builtin-call rewrites (e.g. malloc ->
// side_effect_exprt via goto-programs/builtin_functions.cpp, or sqrtf ->
// ieee_sqrt via clang-c-frontend/clang_c_adjust_expr.cpp) never fire for
// them; the call instead surfaces as a bodyless external function at symex
// time (roadmap §4.8). Recognise the small set of well-known builtins by
// callee name here instead, rewriting the FUNCTION_CALL's `code` into the
// ASSIGN shape the native pipeline would have produced. Returns true if
// `code` was rewritten (the caller must then also override the
// instruction's typeid to ASSIGN).
bool fix_builtin_call(irept &code)
{
  if (code.id() == "nil" || code.find("statement").id() != "function_call")
    return false;

  const irept::subt &sub = code.get_sub();
  if (sub.size() != 3 || sub[1].id() != "symbol")
    return false;

  const std::string callee = sub[1].find("identifier").id_string();

  // memcpy/memset/memmove: CBMC inlines a <builtin-library-*> body that performs
  // the copy via ARRAY_COPY/ARRAY_REPLACE/ARRAY_SET OTHER-instructions, which
  // ESBMC's symex has no handler for and silently skips -- so the copy never
  // happens and a post-copy read of the destination reports a false FAILED.
  // memcmp is a bodyless external returning nondet, so a valid comparison also
  // reports a false FAILED. Rather than teach symex those array ops, retarget
  // the call to ESBMC's own well-tested memory intrinsic: symex dispatches any
  // c:@F@__ESBMC* call to run_intrinsic purely by callee name (symex_main.cpp),
  // so only the function symbol's identifier needs to change -- the 3-argument
  // signature (dst/src/n, s/c/n, s1/s2/n) already matches
  // intrinsic_memcpy/memset/memmove/memcmp, and the lhs may be nil (the return
  // value is often discarded). The instruction stays a FUNCTION_CALL, so return
  // false: the caller then keeps CBMC's original FUNCTION_CALL instruction type
  // rather than forcing it to ASSIGN.
  static const std::unordered_map<std::string, const char *> mem_intrinsics = {
    {"memcpy", "c:@F@__ESBMC_memcpy"},
    {"memset", "c:@F@__ESBMC_memset"},
    {"memmove", "c:@F@__ESBMC_memmove"},
    {"memcmp", "c:@F@__ESBMC_memcmp"}};
  auto mem_it = mem_intrinsics.find(callee);
  if (mem_it != mem_intrinsics.end())
  {
    code.get_sub()[1].set("identifier", mem_it->second);
    return false;
  }

  // Copy out of `code` before mutating it below -- sub/args (and anything
  // referencing into them) alias code.get_sub(), which code.get_sub().clear()
  // invalidates.
  const irept::subt args = sub[2].get_sub();

  // free(ptr) returns void, so unlike the value-returning builtins below it has
  // a nil lhs and lowers to an OTHER instruction carrying a "free" codet (the
  // shape goto_convertt::do_free produces), not an assign. migrate_expr maps
  // that to code_free2t -> symex_free, which actually deallocates and so lets
  // ESBMC detect use-after-free on CBMC binaries (otherwise free is a bodyless
  // external returning nondet and the deallocation is silently dropped).
  if (callee == "free")
  {
    if (args.size() != 1)
      return false;
    const irept ptr = args[0];
    code.set("statement", "free");
    code.get_sub().clear();
    code.get_sub().push_back(ptr);
    return true;
  }

  if (sub[0].is_nil())
    return false; // do_mem/the AST rewrite are themselves no-ops here

  const irept lhs = sub[0];

  irept rhs;
  if (callee == "malloc")
    rhs = build_mem_rhs(lhs, args, "malloc");
  else if (callee == "alloca" || callee == "__builtin_alloca")
    rhs = build_mem_rhs(lhs, args, "alloca");
  else if (callee == "realloc")
    rhs = build_realloc_rhs(lhs, args);
  else if (callee == "sqrtf" || callee == "sqrt" || callee == "sqrtl")
    rhs = build_unary_fp_rhs(lhs, args, "ieee_sqrt");
  else if (
    callee == "nearbyint" || callee == "nearbyintf" || callee == "nearbyintl")
    rhs = build_unary_fp_rhs(lhs, args, "nearbyint");
  else if (callee == "fma" || callee == "fmaf" || callee == "fmal")
    rhs = build_fma_rhs(lhs, args);
  else if (callee == "__builtin_nan" || callee == "__builtin_nanf")
    rhs = build_nan_rhs(lhs);
  else if (
    callee == "__builtin_huge_val" || callee == "__builtin_huge_valf" ||
    callee == "__builtin_huge_vall" || callee == "__builtin_inf" ||
    callee == "__builtin_inff" || callee == "__builtin_infl")
    rhs = build_inf_rhs(lhs);
  // "abs" mirrors what clang_c_adjust_expr.cpp builds for a recognised
  // fabs/fabsf/fabsl call; migrate_expr's abs handler reads op0(), so "abs"
  // must be in fix_expression's operand-wrap set for the argument to reach it.
  // The native abs expr is type-agnostic (build_unary_fp_rhs takes the lhs
  // type), so the same rewrite covers the integer abs family -- CBMC emits
  // abs/labs/llabs/imaxabs (and their __builtin_ spellings) as bodyless
  // FUNCTION_CALL externals too, so without this ESBMC returns nondet and a
  // valid abs(-7)==7 reports FAILED where CBMC says SUCCESSFUL.
  else if (
    callee == "fabsf" || callee == "fabs" || callee == "fabsl" ||
    callee == "abs" || callee == "labs" || callee == "llabs" ||
    callee == "imaxabs" || callee == "__builtin_abs" ||
    callee == "__builtin_labs" || callee == "__builtin_llabs" ||
    callee == "__builtin_imaxabs")
    rhs = build_unary_fp_rhs(lhs, args, "abs");
  else
    return false; // not (yet) a recognised builtin; see roadmap §4.8

  if (rhs.is_nil())
    return false; // wrong arity for the builtin matched above

  code.set("statement", "assign");
  code.get_sub().clear();
  code.get_sub().push_back(lhs);
  code.get_sub().push_back(rhs);
  return true;
}

irept symbol_to_esbmc_irep(const cbmc_symbolt &sym)
{
  irept result;
  result.add("type") = sym.stype;
  result.add("symvalue") = sym.value;
  result.add("location") = sym.location;
  result.add("module") = mk(sym.module);
  result.add("mode") = mk(sym.mode);

  // fix_name is the identity in adapter.rs.
  if (sym.is_type)
    result.add("is_type") = mk("1");
  if (sym.is_macro)
    result.add("is_macro") = mk("1");
  if (sym.is_parameter)
    result.add("is_parameter") = mk("1");
  if (sym.is_lvalue)
    result.add("lvalue") = mk("1");
  if (sym.is_static_lifetime)
    result.add("static_lifetime") = mk("1");
  if (sym.is_file_local)
    result.add("file_local") = mk("1");
  if (sym.is_extern)
    result.add("is_extern") = mk("1");

  // thread_local translates directly: symbolt::is_thread_local is honoured by
  // symex (per-thread L1 renaming, renaming.cpp) and the race analysis
  // (rw_set.cpp). Only the static-lifetime case carries information -- CBMC
  // sets the bit on every stack-allocated symbol (locals, parameters, return
  // slots), which are per-frame in ESBMC by construction, so those are
  // dropped silently rather than warned about on every load.
  if (sym.is_thread_local && sym.is_static_lifetime)
    result.is_thread_local(true);

  // Flags ESBMC has no equivalent for are dropped (roadmap §4.5). Warn when
  // one is actually set: volatile at the symbol level is rare (CBMC keeps the
  // qualifier in the type, which migrates independently), weak matters only
  // if a later link step could override the definition, and property marks
  // CBMC-internal assertion bookkeeping.
  if (sym.is_volatile)
    log_warning("CBMC adapter: dropping 'volatile' on symbol {}", sym.name);
  if (sym.is_weak)
    log_warning("CBMC adapter: dropping 'weak' on symbol {}", sym.name);
  if (sym.is_property)
    log_warning("CBMC adapter: dropping 'property' on symbol {}", sym.name);

  result.add("base_name") = mk(sym.base_name);
  result.add("name") = mk(sym.name);

  fix_expression(result);
  return result;
}

irept instruction_to_esbmc_irep(
  const cbmc_instructiont &ins,
  const std::map<unsigned, unsigned> &target_revmap,
  const std::string &function_name)
{
  irept result;

  // ESBMC expects code arguments inside "operands".
  irept code = ins.code;

  // CBMC's whole-object codet statements have no ESBMC symex counterpart, so
  // migrate would abort() on them (SIGABRT). `array_set` (__CPROVER_array_set:
  // set every element of the pointed-to array) carries no explicit length -- the
  // extent comes from the pointee's type, which ESBMC's memset/array machinery
  // does not reconstruct here -- and `havoc_object` (set the whole pointed-to
  // object nondet) is likewise size-implicit. Decline cleanly (a throw the
  // create_goto_program handler turns into a graceful error exit, roadmap §4.7)
  // rather than crashing; these reach the adapter only from an explicit
  // __CPROVER_array_set / __CPROVER_havoc_object (CBMC's own memset lowering is
  // retargeted to __ESBMC_memset in fix_builtin_call before its ARRAY_SET body
  // runs, §4.8).
  if (code.id() != "nil")
  {
    const irep_idt stmt = code.find("statement").id();
    if (stmt == "array_set" || stmt == "havoc_object")
      throw std::string(
        "CBMC adapter: '" + stmt.as_string() +
        "' whole-object operations are not yet supported on the --binary path");
  }

  const bool rewrote_builtin_call = fix_builtin_call(code);
  irept operands;
  operands.get_sub() = code.get_sub();
  code.get_sub().clear();
  code.add("operands") = operands;

  if (code.id() != "nil" && code.find("statement").id() == "assign")
    if (code.find("operands").get_sub().size() != 2)
    {
      log_error("CBMC adapter: assign must have exactly two operands");
      abort();
    }

  result.add("code") = code;
  result.add("location") = ins.source_location;
  // fix_builtin_call rewrote a FUNCTION_CALL into an ASSIGN (malloc/sqrt/...) or
  // an OTHER "free" codet; the instruction kind must agree with the rewritten
  // code, not CBMC's original raw type. 13 is ASSIGN, 4 is OTHER (shared
  // numbering, see map_cbmc_instruction_type).
  result.add("typeid") = mk(
    rewrote_builtin_call
      ? (code.find("statement").id() == "free" ? "4" : "13")
      : std::to_string(map_cbmc_instruction_type(ins.instr_type)));
  result.add("guard") = ins.guard;

  if (!ins.targets.empty())
  {
    irept t_ireps;
    for (unsigned raw : ins.targets)
    {
      auto it = target_revmap.find(raw);
      if (it == target_revmap.end())
      {
        log_error("CBMC adapter: unresolved target number {}", raw);
        abort();
      }
      t_ireps.get_sub().push_back(mk(std::to_string(it->second)));
    }
    result.add("targets") = t_ireps;
  }

  if (!ins.labels.empty())
  {
    irept l_ireps;
    for (const auto &label : ins.labels)
      l_ireps.get_sub().push_back(mk(label));
    result.add("labels") = l_ireps;
  }

  result.add("function") = mk(function_name);

  fix_expression(result);
  return result;
}

irept function_to_esbmc_irep(const cbmc_functiont &func)
{
  // CBMC numbers targets from 1 with a per-instruction target number; ESBMC
  // uses the instruction's position. Build the reverse map over ALL
  // instructions first (matching adapter.rs), then emit.
  std::map<unsigned, unsigned> target_revmap;
  for (unsigned i = 0; i < func.instructions.size(); i++)
    target_revmap[func.instructions[i].target_number] = i;

  irept result(irep_idt("goto-program"));
  for (const auto &ins : func.instructions)
  {
    const bool is_output =
      ins.code.id() != "nil" && ins.code.find("statement").id() == "output";
    if (!is_output)
      result.get_sub().push_back(
        instruction_to_esbmc_irep(ins, target_revmap, func.name));
  }
  return result;
}

} // namespace

unsigned map_cbmc_instruction_type(unsigned cbmc_type)
{
  // CBMC's goto_program_instruction_typet (CBMC src/goto-programs/goto_program.h),
  // named here so the CBMC->ESBMC mapping is explicit and auditable. The
  // numbering matches CBMC's enum exactly.
  enum cbmc_instruction_typet
  {
    CBMC_NO_INSTRUCTION_TYPE = 0,
    CBMC_GOTO = 1,
    CBMC_ASSUME = 2,
    CBMC_ASSERT = 3,
    CBMC_OTHER = 4,
    CBMC_SKIP = 5,
    CBMC_START_THREAD = 6,
    CBMC_END_THREAD = 7,
    CBMC_LOCATION = 8,
    CBMC_END_FUNCTION = 9,
    CBMC_ATOMIC_BEGIN = 10,
    CBMC_ATOMIC_END = 11,
    CBMC_RETURN = 12, // renamed SET_RETURN_VALUE upstream; value unchanged
    CBMC_ASSIGN = 13,
    CBMC_DECL = 14,
    CBMC_DEAD = 15,
    CBMC_FUNCTION_CALL = 16,
    CBMC_THROW = 17,
    CBMC_CATCH = 18,
    CBMC_INCOMPLETE_GOTO = 19,
  };

  switch (cbmc_type)
  {
  // ESBMC shares these enumerator values (goto_program.h), so the raw CBMC
  // value already names the right ESBMC kind: map by identity.
  case CBMC_NO_INSTRUCTION_TYPE:
  case CBMC_GOTO:
  case CBMC_ASSUME:
  case CBMC_ASSERT:
  case CBMC_OTHER:
  case CBMC_SKIP:
  case CBMC_LOCATION:
  case CBMC_END_FUNCTION:
  case CBMC_ATOMIC_BEGIN:
  case CBMC_ATOMIC_END:
  case CBMC_RETURN:
  case CBMC_ASSIGN:
  case CBMC_DECL:
  case CBMC_DEAD:
  case CBMC_FUNCTION_CALL:
  case CBMC_THROW:
  case CBMC_CATCH:
    return cbmc_type;

  case CBMC_START_THREAD:
  case CBMC_END_THREAD:
    log_error(
      "CBMC adapter: thread instruction ({}) is not supported; ESBMC models "
      "concurrency as intrinsic calls, not goto instruction types",
      cbmc_type == CBMC_START_THREAD ? "START_THREAD" : "END_THREAD");
    abort();

  case CBMC_INCOMPLETE_GOTO:
    log_error(
      "CBMC adapter: INCOMPLETE_GOTO (19) in a finished binary; the goto "
      "target was never resolved");
    abort();

  default:
    log_error("CBMC adapter: unknown CBMC instruction type {}", cbmc_type);
    abort();
  }
}

cbmc_adapted_resultt adapt_cbmc_to_esbmc(cbmc_parse_resultt parsed)
{
  cbmc_adapted_resultt out;
  out.symbols.reserve(parsed.symbols.size());
  out.functions.reserve(parsed.functions.size());

  // First, map all struct ref-types into concrete types via a tag cache.
  std::unordered_map<std::string, irept> type_cache;

  for (auto &sym : parsed.symbols)
  {
    if (
      sym.is_type && (sym.stype.id() == "struct" || sym.stype.id() == "union"))
    {
      // A struct_tag/union_tag reference identifies its definition by the type
      // symbol's *name*, which is scope-qualified: `tag-S` at file scope but
      // `main::1::tag-S` for a struct declared inside a function body. Keying
      // the cache by `"tag-" + base_name` only matched the former, so any
      // function-local struct went unresolved and aborted ("struct_tag/union_tag
      // should have been resolved"). Key by the symbol name, which equals the
      // reference identifier at every scope (identical to the old key at file
      // scope, where name == "tag-" + base_name).
      // Seed the expansion stack with the definition's own identity so
      // self-references stay "symbol" back-references (see
      // fix_type_symbol_definition).
      fix_type_symbol_definition(sym.stype, type_cache, sym.name);
      type_cache[sym.name] = sym.stype;
    }
    out.symbols.push_back(symbol_to_esbmc_irep(sym));
  }

  // A symbol might be defined later; re-check every symbol with the full cache.
  for (auto &symbol : out.symbols)
  {
    const bool is_type_symbol = has_sub(symbol, "is_type");
    if (is_type_symbol)
      fix_type_symbol_definition(
        symbol, type_cache, symbol.find("name").id_string());
    else
      fix_type(symbol, type_cache);
    if (
      irep_contains(symbol, "struct_tag") || irep_contains(symbol, "union_tag"))
    {
      log_error("CBMC adapter: struct_tag/union_tag should have been resolved");
      abort();
    }
    if (symbol.find("type").id() == "c_bool")
    {
      log_error("CBMC adapter: c_bool type should have been rewritten");
      abort();
    }
  }

  for (auto &func : parsed.functions)
  {
    irept function_irep = function_to_esbmc_irep(func);
    fix_type(function_irep, type_cache);
    out.functions.emplace_back(func.name, std::move(function_irep));
  }

  return out;
}
