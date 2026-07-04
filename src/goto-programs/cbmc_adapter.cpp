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

// CBMC sometimes stores constant values as hex; ESBMC wants a binary string.
// Mirrors Rust `format!("{:032b}", u64::from_str_radix(value, 16))`.
std::string hex_to_bin32(const std::string &hex)
{
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
  if (bits.size() < 32)
    bits = std::string(32 - bits.size(), '0') + bits;
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

void fix_expression(irept &irep)
{
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
      if (val.size() != 32)
        irep.add("value") = mk(hex_to_bin32(val));
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
    "abs"};

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
  log_error("CBMC adapter: unsupported anonymous struct {}", ident);
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

  if (self.id() != "struct_tag")
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

  // The resolved struct may itself contain struct tags; redo the cache walk.
  if (irep_contains(self, "struct_tag"))
  {
    for (auto &v : self.get_sub())
      fix_type(v, cache);
    for (auto &p : self.get_named_sub())
      fix_type(p.second, cache);
    for (auto &p : self.get_comments())
      fix_type(p.second, cache);
  }
}

// Builds the malloc side_effect_exprt (irep shape) do_mem would have built,
// falling back to element type char -- do_mem's own fallback whenever the
// size argument isn't a recognisable sizeof(T) pattern, which a CBMC
// binary's argument never is (sizeof is constant-folded away by goto-cc
// well before the .goto file exists). A byte-granularity allocation is a
// sound, if less precise, model of any malloc(n) call regardless of the
// pointer type it's later cast to.
irept build_malloc_rhs(const irept &lhs, const irept::subt &args)
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
  sideeffect.add("statement") = mk("malloc");
  sideeffect.get_sub().push_back(args[0]);
  sideeffect.add("#size") = size_arg;
  sideeffect.add("#type") = static_cast<const irept &>(char_type());
  return sideeffect;
}

// Builds an ieee_sqrt exprt (irep shape), mirroring what
// clang_c_adjust_expr.cpp builds for a syntactically-recognised sqrt/sqrtf/
// sqrtl call. No explicit rounding_mode operand -- migrate_expr's ieee_sqrt
// handler defaults to the standard c:@__ESBMC_rounding_mode symbol when one
// isn't present, same as it does for the ieee_add/sub/mul/div family. That
// default symbol is defined by esbmc_parseoptions.cpp's
// synthesize_cprover_additions, which runs before read_cbmc_goto_object in
// every *normal* --binary invocation, but not under --no-cprover-additions
// -- currently masked there by an unrelated, pre-existing entry-point
// resolution gap, not a live bug today, but not this function's to assume
// away either.
irept build_sqrt_rhs(const irept &lhs, const irept::subt &args)
{
  if (args.size() != 1)
    return get_nil_irep();

  irept result(irep_idt("ieee_sqrt"));
  result.add("type") = lhs.find("type");
  result.get_sub().push_back(args[0]);
  return result;
}

// Builds an "abs" exprt (irep shape), mirroring what
// clang_c_adjust_expr.cpp builds for a syntactically-recognised
// fabs/fabsf/fabsl call. Same unary shape as build_sqrt_rhs; migrate_expr's
// abs handler reads op0(), so "abs" must be in fix_expression's operand-wrap
// set for the argument to reach it.
irept build_abs_rhs(const irept &lhs, const irept::subt &args)
{
  if (args.size() != 1)
    return get_nil_irep();

  irept result(irep_idt("abs"));
  result.add("type") = lhs.find("type");
  result.get_sub().push_back(args[0]);
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

  if (sub[0].is_nil())
    return false; // do_mem/the AST rewrite are themselves no-ops here

  const std::string callee = sub[1].find("identifier").id_string();
  // Copy out of `code` before mutating it below -- sub/args (and anything
  // referencing into them) alias code.get_sub(), which code.get_sub().clear()
  // invalidates.
  const irept lhs = sub[0];
  const irept::subt args = sub[2].get_sub();

  irept rhs;
  if (callee == "malloc")
    rhs = build_malloc_rhs(lhs, args);
  else if (callee == "sqrtf" || callee == "sqrt" || callee == "sqrtl")
    rhs = build_sqrt_rhs(lhs, args);
  else if (callee == "fabsf" || callee == "fabs" || callee == "fabsl")
    rhs = build_abs_rhs(lhs, args);
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
  // fix_builtin_call rewrote a FUNCTION_CALL into an ASSIGN; the instruction
  // kind must agree with the rewritten code, not CBMC's original raw type.
  // 13 is goto_program_instruction_typet::ASSIGN (shared numbering, see
  // map_cbmc_instruction_type).
  result.add("typeid") = mk(
    rewrote_builtin_call
      ? "13"
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
    if (sym.is_type && sym.stype.id() == "struct")
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
    if (irep_contains(symbol, "struct_tag"))
    {
      log_error("CBMC adapter: struct_tag should have been resolved");
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
