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

// CBMC stores integer constant values as hex; ESBMC wants a binary string
// whose length matches the constant's own type width. The Rust reference
// (`format!("{:032b}", ...)`) hardcoded 32, which truncated the representation
// of constants in wider (e.g. 64-bit) types: a 64-bit value was emitted as a
// <=33-char string and silently interpreted at 32 bits, so e.g. -5000000000LL
// verified as its low 32 bits (roadmap §4.3/§7). Pad to `width` bits instead.
// Values needing more than 64 bits (128-bit constants, §4.3) are out of range
// for this uint64_t path and are returned unchanged rather than crashing
// std::stoull -- note this leaves such a value as a raw hex string (a known
// limitation, roadmap §4.3), but the >64-bit path is not otherwise exercised.
std::string hex_to_bin(const std::string &hex, std::size_t width)
{
  if (hex.size() > 16) // > 64 bits: cannot round-trip through uint64_t
    return hex;
  unsigned long long n = std::stoull(hex, nullptr, 16);
  std::string bits;
  if (n == 0)
    bits = "0";
  else
    while (n != 0)
    {
      bits.push_back(static_cast<char>('0' + (n & 1)));
      n >>= 1;
    }
  std::reverse(bits.begin(), bits.end());
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

void fix_expression(irept &irep)
{
  if (
    irep.id() == "count_leading_zeros" ||
    irep.id() == "count_trailing_zeros")
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
      // Value may be a hex representation or binary; we want the binary one.
      // A value already exactly 32 chars is treated as an existing 32-bit
      // binary string and left as-is; anything else is a hex value converted to
      // a binary string of the type's own bit width (see hex_to_bin).
      if (val.size() != 32)
        irep.add("value") = mk(hex_to_bin(val, bv_width(irep.find("type"))));
    }
  }

  static const std::unordered_set<std::string> expressions = {
    "if",
    "member",
    "typecast",
    "notequal",
    "and",
    "or",
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
    "bswap"};

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

void expand_anon_struct(const irept &self)
{
  if (has_sub(self, "components"))
    return;
  // ESBMC has no parser for CBMC's anonymous naming convention.
  const std::string ident = self.find("identifier").id_string();
  if (ident.size() < 11 || ident.compare(0, 10, "tag-#anon#") != 0)
    return;
  log_error("CBMC adapter: unsupported anonymous aggregate {}", ident);
  abort();
}

void fix_type(irept &self, const std::unordered_map<std::string, irept> &cache)
{
  if (self.id() == "c_bool")
  {
    self.id("signedbv");
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

  if (
    self.id() == "pointer" && !has_sub(self, "subtype") &&
    !self.get_sub().empty())
  {
    for (auto &v : self.get_sub())
      fix_type(v, cache);
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

  // struct_tag and union_tag are the two aggregate references CBMC emits; both
  // resolve out of the same tag cache into their concrete definition.
  const bool is_tag = self.id() == "struct_tag" || self.id() == "union_tag";
  if (!is_tag)
  {
    for (auto &v : self.get_sub())
      fix_type(v, cache);
    for (auto &p : self.get_named_sub())
      fix_type(p.second, cache);
    for (auto &p : self.get_comments())
      fix_type(p.second, cache);
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

  self = it->second;

  // The resolved aggregate may itself contain tags; redo the cache walk.
  if (irep_contains(self, "struct_tag") || irep_contains(self, "union_tag"))
  {
    for (auto &v : self.get_sub())
      fix_type(v, cache);
    for (auto &p : self.get_named_sub())
      fix_type(p.second, cache);
    for (auto &p : self.get_comments())
      fix_type(p.second, cache);
  }
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

  // Flags ESBMC has no equivalent for are dropped (roadmap §4.5). Warn when one
  // is actually set, since volatile/thread_local in particular affect soundness.
  if (sym.is_volatile)
    log_warning("CBMC adapter: dropping 'volatile' on symbol {}", sym.name);
  if (sym.is_thread_local)
    log_warning("CBMC adapter: dropping 'thread_local' on symbol {}", sym.name);
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
      const std::string tagname = "tag-" + sym.base_name;
      fix_type(sym.stype, type_cache);
      type_cache[tagname] = sym.stype;
    }
    out.symbols.push_back(symbol_to_esbmc_irep(sym));
  }

  // A symbol might be defined later; re-check every symbol with the full cache.
  for (auto &symbol : out.symbols)
  {
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
