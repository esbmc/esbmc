#include <goto-symex/foundry.h>
#include <goto-symex/slice.h>
#include <ac_config.h>
#include <util/prefix.h>
#include <util/mp_arith.h>
#include <util/message/format.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <irep2/irep2_expr.h>
#include <fstream>
#include <functional>
#include <sstream>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cctype>

// ---------------------------------------------------------------------------
// Symbol-name parsing
// ---------------------------------------------------------------------------

bool foundry_generator::parse_param_symbol(
  const std::string &raw,
  std::string &contract,
  std::string &method,
  std::string &param)
{
  // Strip SSA level/renaming suffixes: keep the base identifier only.
  std::string name = raw;
  size_t cut = name.find_first_of("?!&#");
  if (cut != std::string::npos)
    name.resize(cut);

  // Solidity function parameters are named
  // `sol:@C@<contract>@F@<method>@<param>`.
  const std::string pfx = "sol:@C@";
  if (!has_prefix(name, pfx))
    return false;
  std::string rest = name.substr(pfx.size());

  size_t f = rest.find("@F@");
  if (f == std::string::npos)
    return false;
  contract = rest.substr(0, f);
  std::string after = rest.substr(f + 3);

  size_t a = after.find('@');
  if (a == std::string::npos)
    return false;
  method = after.substr(0, a);
  param = after.substr(a + 1);

  // A genuine parameter is a single trailing identifier (no further scope
  // separators). Locals/temporaries carry extra `@`, contract fields do not
  // reach here (they are not `@F@`-scoped).
  if (
    param.empty() || param.find('@') != std::string::npos || contract.empty() ||
    method.empty())
    return false;
  return true;
}

// ---------------------------------------------------------------------------
// Value formatting
// ---------------------------------------------------------------------------

namespace
{
// The effective Solidity type string of a lowered parameter *type*. A fixed
// bytesN lowers to the `BytesStatic` struct (its `#sol_type` is the tag
// "BytesStatic"); when the source width survives as `#sol_bytesn_size`
// (set in get_elementary_type_name, solidity_convert_type.cpp) it is recovered
// as "BYTES<N>" so the value formatter can render the exact-width literal. All
// other types pass their `#sol_type` through unchanged. The width irep is often
// stripped from the type by type2tc migration before the generator runs; for
// that reason the authoritative source is the width stamped on the code_typet
// argument itself (see arg_sol_type / get_function_params), not this type.
std::string effective_sol_type(const typet &t)
{
  // A dynamic/fixed array lowers to `pointer<elem>` with the array kind stamped
  // on the POINTER (#sol_type = DYNARRAY/ARRAY/…) and the element type on its
  // subtype. Detect it BEFORE the pointer-deref below (which would otherwise
  // return the element type, mistaking `T[]` for a scalar `T`). Encode as
  // "ARRAY:<elem>" so the array is rendered as a `new <elem>[](N)` literal.
  if (t.is_pointer())
  {
    const std::string ptag = t.get("#sol_type").as_string();
    if (
      ptag == "DYNARRAY" || ptag == "ARRAY" || ptag == "ARRAY_LITERAL" ||
      ptag == "ARRAY_CALLOC")
    {
      const std::string elem = effective_sol_type(t.subtype());
      if (!elem.empty())
        return "ARRAY:" + elem;
    }
    // An interface/contract-typed value lowers to `pointer<struct tag-I>` with
    // `#sol_type: CONTRACT` and `#sol_contract: I` on the pointer. Encode as
    // "CONTRACT:<I>" so the reconstructor can synthesize a mock instance (a bare
    // address literal would revert when the contract calls a method on it).
    if (ptag == "CONTRACT")
    {
      const std::string cn = t.get("#sol_contract").as_string();
      if (!cn.empty())
        return "CONTRACT:" + cn;
    }
    // A `string` value lowers to a `char *` with `#sol_type: STRING` stamped on
    // the pointer. Surface it so a string call-argument is rendered as a
    // Solidity string literal reconstructed from the counterexample length
    // (see format_sol_value / reconstruct's recovered_str_len) instead of being
    // dropped as unsupported.
    if (ptag == "STRING")
      return "STRING";
  }
  // A `bytes` value lowers to the `BytesDynamic` struct (tag "struct
  // BytesDynamic", no #sol_type). NOT "BytesStatic" (a fixed bytesN, rendered
  // from its width elsewhere).
  if (
    t.is_struct() &&
    t.tag().as_string().find("BytesDynamic") != std::string::npos)
    return "BYTES_DYN";

  const typet &ty = t.is_pointer() ? t.subtype() : t;
  // A fixed bytesN carries its width as `#sol_bytesn_size`. Key off that
  // directly — the `#sol_type` tag ("BytesStatic") is not always present on the
  // lowered type across solc AST shapes, but the width is authoritative when
  // set.
  std::string base;
  std::string sz = ty.get("#sol_bytesn_size").as_string();
  if (!sz.empty())
  {
    unsigned n = static_cast<unsigned>(std::stoul(sz));
    if (n >= 1 && n <= 32)
      base = "BYTES" + std::to_string(n);
  }
  if (base.empty())
    base = ty.get("#sol_type").as_string();
  // A user-defined value type (`type Name is <underlying>;`) lowers to its
  // underlying type but is NOT implicitly convertible from a bare underlying
  // literal at a call site. Surface it as "UDVT:<Name>:<underlying>" so the
  // value formatter emits `Name.wrap(<literal>)`.
  std::string udvt = ty.get("#sol_udvt_name").as_string();
  if (!udvt.empty() && !base.empty())
    return "UDVT:" + udvt + ":" + base;
  return base;
}

// A `#sol_type` like "BYTES32" names a fixed-size bytesN (N in 1..32).
// "BytesDynamic"/"BytesStatic" (mixed case) are NOT fixed bytesN and must not
// match. Returns N, or 0 if `sol_type` is not a fixed bytesN.
unsigned parse_fixed_bytes_width(const std::string &sol_type)
{
  if (!has_prefix(sol_type, "BYTES") || sol_type.size() <= 5)
    return 0;
  if (!std::isdigit(static_cast<unsigned char>(sol_type[5])))
    return 0;
  unsigned n = static_cast<unsigned>(std::stoul(sol_type.substr(5)));
  return (n >= 1 && n <= 32) ? n : 0;
}

// Render a `bytesN` value recovered from the model as a Solidity literal
// `bytesN(0x..)`. bytesN is modeled as a BytesStatic struct
// { unsigned char data[32]; size_t length; } with `data` big-endian
// (data[0] = most-significant byte; see bytes_static_from_uint), which is
// exactly how solc lays out a bytesN literal, so emitting data[0..N-1] as hex
// round-trips to the same 32-byte value. Returns "" if the struct/array shape
// is not a fully-concrete constant.
std::string format_fixed_bytes(unsigned n, const expr2tc &value)
{
  if (!is_constant_struct2t(value))
    return "";
  const constant_struct2t &st = to_constant_struct2t(value);
  if (st.datatype_members.empty())
    return "";
  const expr2tc &data = st.datatype_members[0]; // the `data[32]` array
  if (!data) // an unconstrained (e.g. nested-struct) bytesN member — degrade
    return "";

  static const char *hexd = "0123456789abcdef";
  std::string hex;
  auto append_byte = [&hex](const expr2tc &e) -> bool {
    if (!is_constant_int2t(e))
      return false;
    unsigned b =
      static_cast<unsigned>(to_constant_int2t(e).value.to_uint64() & 0xffu);
    hex.push_back(hexd[(b >> 4) & 0xf]);
    hex.push_back(hexd[b & 0xf]);
    return true;
  };

  if (is_constant_array2t(data))
  {
    const constant_array2t &arr = to_constant_array2t(data);
    for (unsigned i = 0; i < n; ++i)
    {
      if (
        i >= arr.datatype_members.size() ||
        !append_byte(arr.datatype_members[i]))
        return "";
    }
  }
  else if (is_constant_array_of2t(data))
  {
    // Whole array collapsed to one repeated element (e.g. all-zero).
    const expr2tc &init = to_constant_array_of2t(data).initializer;
    for (unsigned i = 0; i < n; ++i)
      if (!append_byte(init))
        return "";
  }
  else
    return "";

  return "bytes" + std::to_string(n) + "(0x" + hex + ")";
}

// Effective sol-type string of a code_typet argument. Prefers the fixed-bytes
// width stamped directly on the argument (`#sol_bytesn_size`, set in
// get_function_params) — this is the authoritative source-declared width and
// survives the type migration that strips it from the argument *type*. Falls
// back to the argument type's own tags. `arg` is a code_typet::argumentt
// (an exprt subtype), so both the direct irep and its type are queryable.
std::string arg_sol_type(const exprt &arg)
{
  std::string base;
  const std::string bn = arg.get("#sol_bytesn_size").as_string();
  if (!bn.empty() && bn.find_first_not_of("0123456789") == std::string::npos)
  {
    unsigned n = static_cast<unsigned>(std::stoul(bn));
    if (n >= 1 && n <= 32)
      base = "BYTES" + std::to_string(n);
  }
  // No direct bytesN width stamp: effective_sol_type already applies any UDVT.
  if (base.empty())
    return effective_sol_type(arg.type());
  // A bytesN width was stamped directly on the argument (see
  // solidity_convert_stmt.cpp). Still apply a UDVT wrapper when the parameter
  // is a user-defined value type over bytesN, so we emit `Name.wrap(...)`
  // rather than a bare bytesN literal (not assignable to the UDVT parameter).
  // The UDVT name is likewise stamped on the argument (the bytesN->BytesStatic
  // migration drops it from the type), so read it from the argument first.
  std::string udvt = arg.get("#sol_udvt_name").as_string();
  if (udvt.empty())
  {
    const typet &ty =
      arg.type().is_pointer() ? arg.type().subtype() : arg.type();
    udvt = ty.get("#sol_udvt_name").as_string();
  }
  return udvt.empty() ? base : "UDVT:" + udvt + ":" + base;
}

// Solidity source type name for a `#sol_type` string, used to render array
// element types (`new <name>[](N)`). Recurses through "ARRAY:" for
// arrays-of-arrays. Returns "" for a type we cannot name.
std::string sol_type_to_solidity(const std::string &st)
{
  if (has_prefix(st, "ARRAY:"))
  {
    const std::string e = sol_type_to_solidity(st.substr(6));
    return e.empty() ? "" : e + "[]";
  }
  if (has_prefix(st, "UDVT:"))
  {
    const std::string rest = st.substr(5);
    return rest.substr(0, rest.find(':')); // the UDVT name is itself a type
  }
  if (st == "ADDRESS" || st == "ADDRESS_PAYABLE")
    return "address";
  if (st == "BOOL")
    return "bool";
  if (has_prefix(st, "UINT"))
    return "uint" + (st.size() > 4 ? st.substr(4) : std::string("256"));
  if (has_prefix(st, "INT"))
    return "int" + (st.size() > 3 ? st.substr(3) : std::string("256"));
  if (unsigned n = parse_fixed_bytes_width(st))
    return "bytes" + std::to_string(n);
  if (st == "BYTES_DYN")
    return "bytes";
  if (st == "STRING")
    return "string";
  return "";
}
} // namespace

std::string foundry_generator::format_sol_value(
  const std::string &sol_type,
  const expr2tc &value)
{
  // User-defined value type "UDVT:<Name>:<underlying>": render the underlying
  // literal, then wrap it (`Name.wrap(<literal>)`) — the only assignable form.
  if (has_prefix(sol_type, "UDVT:"))
  {
    const std::string rest = sol_type.substr(5);
    size_t sep = rest.find(':');
    if (sep == std::string::npos)
      return "";
    const std::string name = rest.substr(0, sep);
    const std::string inner = format_sol_value(rest.substr(sep + 1), value);
    return inner.empty() ? "" : name + ".wrap(" + inner + ")";
  }

  if (sol_type == "BOOL")
  {
    if (is_constant_bool2t(value))
      return to_constant_bool2t(value).value ? "true" : "false";
    if (is_constant_int2t(value))
      return (to_constant_int2t(value).value != 0) ? "true" : "false";
    return "";
  }

  // Fixed-size bytesN (bytes1..bytes32): recovered as a BytesStatic struct.
  // The width N MUST come from the declared source type (`#sol_type` =
  // "BYTES<N>"); we never infer it from the recovered struct's `.length`, which
  // is a free nondet value on any path that does not constrain the bytesN and
  // could otherwise yield a wrong-width literal for the declared parameter. If
  // the caller cannot supply the declared width, the value degrades to "" (the
  // call is then UNSUPPORTED) rather than risk a wrong-width/wrong-value test.
  if (unsigned n = parse_fixed_bytes_width(sol_type))
    return format_fixed_bytes(n, value);

  // Dynamic `bytes`: recovered as a BytesDynamic struct
  // {offset,length,capacity,initialized,…}. The byte CONTENT lives in a
  // separate pool addressed by `offset` and is not faithfully recoverable, but
  // the `.length` field is — and a branch on a `bytes` argument reads it via
  // `.length`. Render a zero-filled literal of the recovered length so both a
  // `d.length > k` arm and its complement get a genuinely length-correct,
  // reaching argument. A garbage-huge nondet length (llc_nondet_bytes leaves
  // the length unconstrained above, so the solver may pick e.g. 2^64-4 to
  // satisfy `> k`) is clamped to a small representative that still exceeds the
  // common small thresholds — a faithful large-content witness is not
  // reconstructible, and an empty default would silently claim the branch
  // without reaching it.
  if (sol_type == "BYTES_DYN")
  {
    if (
      !is_constant_struct2t(value) ||
      !is_struct_type(to_constant_struct2t(value).type))
      return "";
    const constant_struct2t &cs = to_constant_struct2t(value);
    const struct_type2t &st = to_struct_type(cs.type);
    size_t idx = st.member_names.size();
    for (size_t i = 0; i < st.member_names.size(); ++i)
      if (st.member_names[i].as_string() == "length")
      {
        idx = i;
        break;
      }
    if (idx >= cs.datatype_members.size())
      return "";
    const expr2tc &lm = cs.datatype_members[idx];
    if (!lm || !is_constant_int2t(lm))
      return "";
    uint64_t len = to_constant_int2t(lm).value.to_uint64();
    // > 4096 is treated as an unconstrained-nondet "garbage" length.
    unsigned long render_len = (len > 4096) ? 32ul : (unsigned long)len;
    return "hex\"" + std::string(2 * render_len, '0') + "\"";
  }

  if (!is_constant_int2t(value))
    return "";
  const BigInt &n = to_constant_int2t(value).value;

  // Unsigned / signed integers: a decimal literal is always valid and exact.
  if (has_prefix(sol_type, "UINT") || has_prefix(sol_type, "INT"))
    return integer2string(n);

  // Address: go through uint160 so we never emit a 40-hex-digit literal
  // (which Solidity rejects unless EIP-55 checksummed).
  if (sol_type == "ADDRESS" || sol_type == "ADDRESS_PAYABLE")
    return "address(uint160(" + integer2string(n) + "))";

  // bytesN / string / aggregates: not faithfully renderable yet -> caller
  // marks the call unsupported rather than emit a wrong literal.
  return "";
}

std::string foundry_generator::default_sol_literal(const std::string &sol_type)
{
  if (has_prefix(sol_type, "UDVT:"))
  {
    const std::string rest = sol_type.substr(5);
    size_t sep = rest.find(':');
    if (sep == std::string::npos)
      return "";
    const std::string inner = default_sol_literal(rest.substr(sep + 1));
    return inner.empty() ? "" : rest.substr(0, sep) + ".wrap(" + inner + ")";
  }
  if (sol_type == "BOOL")
    return "false";
  if (has_prefix(sol_type, "UINT") || has_prefix(sol_type, "INT"))
    return "0";
  if (sol_type == "ADDRESS" || sol_type == "ADDRESS_PAYABLE")
    return "address(0)";
  // Fixed-size bytesN not exercised on the path: the zero value is a valid,
  // faithful default (a sliced/unread bytesN cannot change branch reachability).
  if (unsigned n = parse_fixed_bytes_width(sol_type))
    return "bytes" + std::to_string(n) + "(0x" + std::string(2 * n, '0') + ")";
  // Dynamic array `T[]`: render a `new <T>[](N)` literal. N mirrors the
  // external-call harness's fixed dynamic-array length (kHarnessDynLen = 4 in
  // solidity_convert_call.cpp), so length-dependent branches (`arr.length`,
  // loop-enter) are reached the same way ESBMC's model reached them. Elements
  // are zero-initialised. A memory array literal binds to a calldata parameter.
  if (has_prefix(sol_type, "ARRAY:"))
  {
    const std::string elem = sol_type_to_solidity(sol_type.substr(6));
    return elem.empty() ? "" : "new " + elem + "[](4)";
  }
  // `bytes` / `string`: an empty literal is a valid, compilable default. Sound
  // when the value is sliced; the try/catch wrap tolerates any content-dependent
  // divergence.
  if (sol_type == "BYTES_DYN")
    return "hex\"\"";
  if (sol_type == "STRING")
    return "\"\"";
  return "";
}

// A struct type NAME (possibly "struct <tag>") that is the modeled bytes value
// wrapper — `BytesStatic` (fixed bytesN) or `BytesDynamic` (`bytes`) — which have
// their own renderers and must NOT be treated as a user struct. Matched exactly
// (not by substring) so a legitimately-named user struct like `BytesBundle` is
// not misclassified.
static bool is_bytes_wrapper_struct(const std::string &name)
{
  std::string n = name;
  if (has_prefix(n, "struct "))
    n = n.substr(7);
  return n == "BytesStatic" || n == "BytesDynamic";
}

std::string foundry_generator::format_struct_literal(
  const namespacet &ns,
  const expr2tc &value,
  std::string &qualified,
  std::set<std::string> &out_imports)
{
  if (!value || !is_constant_struct2t(value))
    return "";
  const constant_struct2t &cs = to_constant_struct2t(value);
  if (!is_struct_type(cs.type))
    return "";

  // The recovered value's migrated type keeps only the LOCAL struct name; the
  // DECLARED tag symbol `tag-struct <Qualified>` retains the source field types
  // and the Solidity-qualified name. Match the tag by local name, requiring
  // UNIQUENESS — if two structs share a local name across scopes (`I1.S` /
  // `I2.S`), we cannot tell which the parameter is (the declared parameter type
  // is an inline struct that also carries only the local name), so degrade
  // rather than pick the wrong one (an uncompilable / wrong-typed literal).
  const std::string tag_pfx = "tag-struct ";
  std::string local = to_struct_type(cs.type).name.as_string();
  if (has_prefix(local, "struct "))
    local = local.substr(7);
  const symbolt *tag = nullptr;
  unsigned matches = 0;
  ns.get_context().foreach_operand([&](const symbolt &s) {
    if (!s.type.is_struct() || !has_prefix(s.id.as_string(), tag_pfx))
      return;
    const std::string q = s.id.as_string().substr(tag_pfx.size());
    // Match the local name exactly, or as the `.<local>` suffix of a
    // contract-scoped qualified name (`IBaseEscrow.Immutables`).
    if (
      q == local ||
      (q.size() > local.size() + 1 &&
       q.compare(q.size() - local.size(), local.size(), local) == 0 &&
       q[q.size() - local.size() - 1] == '.'))
    {
      ++matches;
      tag = &s;
    }
  });
  if (matches != 1)
    return ""; // ambiguous or not found — degrade
  qualified = tag->id.as_string().substr(tag_pfx.size());
  // Import the struct's own scope (`Scope.Name` → `Scope`; top-level → `Name`).
  {
    size_t dot = qualified.find('.');
    out_imports.insert(
      dot == std::string::npos ? qualified : qualified.substr(0, dot));
  }

  const struct_typet &decl = to_struct_type(tag->type);
  const auto &comps = decl.components();
  // Declared components (incl. padding) must align 1:1 with recovered members.
  if (comps.size() != cs.datatype_members.size())
    return "";

  std::string out;
  for (size_t i = 0; i < comps.size(); ++i)
  {
    const std::string cname = comps[i].get_name().as_string();
    if (has_prefix(cname, "anon_pad"))
      continue; // synthetic padding — not a source field
    if (!cs.datatype_members[i])
      return ""; // unrecovered member — degrade rather than deref null
    const typet &ft = comps[i].type();
    std::string fsol = effective_sol_type(ft);
    // A fixed `bytesN` field lost its width to the type-follow; the frontend
    // re-stamps it on the component irep (get_struct_class_fields), which
    // survives. Recover the exact-width sol-type from there.
    if (fsol.empty())
    {
      const std::string bn = comps[i].get("#sol_bytesn_size").as_string();
      if (
        !bn.empty() && bn.find_first_not_of("0123456789") == std::string::npos)
      {
        unsigned n = static_cast<unsigned>(std::stoul(bn));
        if (n >= 1 && n <= 32)
          fsol = "BYTES" + std::to_string(n);
      }
    }
    // A fixed/dynamic array struct field cannot be rendered faithfully inside a
    // positional literal (dynamic default `new T[](4)` is illegal for a fixed
    // field) — degrade the whole struct rather than emit a wrong literal.
    if (has_prefix(fsol, "ARRAY:") || ft.is_array())
      return "";
    // A nested USER struct field recurses; a bytes field is ALSO a struct
    // (BytesStatic/BytesDynamic) but must render via format_sol_value, not
    // recursion — distinguish by the member's struct tag.
    bool member_user_struct = false;
    if (is_constant_struct2t(cs.datatype_members[i]))
    {
      const type2tc &mt = cs.datatype_members[i]->type;
      member_user_struct =
        is_struct_type(mt) &&
        !is_bytes_wrapper_struct(to_struct_type(mt).name.as_string());
    }
    std::string lit;
    if (member_user_struct)
    {
      std::string inner_q;
      lit =
        format_struct_literal(ns, cs.datatype_members[i], inner_q, out_imports);
    }
    if (lit.empty())
      lit = format_sol_value(fsol, cs.datatype_members[i]);
    if (lit.empty())
      lit = default_sol_literal(fsol);
    if (lit.empty())
      return ""; // all-or-nothing
    // A UDVT field renders as `Name.wrap(...)`; import the top-level UDVT type.
    if (has_prefix(fsol, "UDVT:"))
    {
      std::string un = fsol.substr(5);
      un = un.substr(0, un.find(':'));
      if (un.find('.') == std::string::npos)
        out_imports.insert(un);
    }
    out += (out.empty() ? "" : ", ") + lit;
  }
  return qualified + "(" + out + ")";
}

const std::vector<std::pair<std::string, std::string>> &
foundry_generator::get_method_params(
  const namespacet &ns,
  const std::string &contract,
  const std::string &method) const
{
  std::string key = contract + "@" + method;
  auto it = method_params.find(key);
  if (it != method_params.end())
    return it->second;

  std::vector<std::pair<std::string, std::string>> params;
  const std::string fn_prefix = "sol:@C@" + contract + "@F@" + method;
  const symbolt *fn = nullptr;
  ns.get_context().foreach_operand([&](const symbolt &s) {
    if (fn || !s.type.is_code())
      return;
    const std::string id = s.id.as_string();
    if (!has_prefix(id, fn_prefix))
      return;
    // The function symbol is `<prefix>#<node-id>`; a parameter/local is
    // `<prefix>@<name>`. Reject the latter.
    if (id.size() > fn_prefix.size() && id[fn_prefix.size()] != '#')
      return;
    fn = &s;
  });

  if (fn)
  {
    const code_typet &ct = to_code_type(fn->type);
    for (const auto &arg : ct.arguments())
    {
      // The first argument of every contract method is the `this`
      // self-pointer synthesised by the frontend; it is not a source-level
      // parameter and must not be emitted.
      std::string pname = arg.get_base_name().as_string();
      if (pname.empty() || pname == "this")
        continue;
      // Source type: prefer the width stamped on the argument (arg_sol_type,
      // which also handles bytesN via `#sol_bytesn_size`), then the parameter
      // symbol (`sol:@C@<C>@F@<method>@<param>`) if the argument carries none.
      std::string stype = arg_sol_type(arg);
      if (stype.empty())
      {
        std::string pid = fn_prefix + "@" + pname;
        if (const symbolt *ps = ns.lookup(irep_idt(pid)))
          stype = ps->type.get("#sol_type").as_string();
      }
      params.emplace_back(pname, stype);
    }
  }

  return method_params.emplace(key, std::move(params)).first->second;
}

// Recurse a dispatcher body, recording every direct call to a contract method
// `sol:@C@<C>@F@<base>#<id>` (prefix `cpfx`) as base -> {ids}. Aux/modifier
// helpers the dispatcher never calls directly are naturally excluded.
static void collect_dispatch_calls(
  const exprt &e,
  const std::string &cpfx,
  std::map<std::string, std::vector<std::string>> &calls)
{
  // The frontend emits calls as `side_effect_expr_function_call` (id
  // "sideeffect", statement "function_call"); goto-lowered bodies use
  // `code_function_call`. Handle both; the callee is the function operand.
  const exprt *callee = nullptr;
  if (e.is_code() && to_code(e).get_statement() == "function_call")
    callee = &to_code_function_call(to_code(e)).function();
  else if (
    e.id() == "sideeffect" && e.get("statement") == "function_call" &&
    !e.operands().empty())
    callee = &e.op0();

  if (callee && callee->id() == "symbol")
  {
    const std::string id = callee->get("identifier").as_string();
    if (has_prefix(id, cpfx))
    {
      std::string rest = id.substr(cpfx.size());
      size_t h = rest.find('#');
      std::string base = h == std::string::npos ? rest : rest.substr(0, h);
      if (
        !base.empty() && base[0] != '_' && base[0] != '$' &&
        base.find('@') == std::string::npos)
      {
        auto &v = calls[base];
        if (std::find(v.begin(), v.end(), id) == v.end())
          v.push_back(id);
      }
    }
  }
  forall_operands (op, e)
    collect_dispatch_calls(*op, cpfx, calls);
}

const std::map<std::string, std::vector<std::string>> &
foundry_generator::dispatcher_callable(
  const namespacet &ns,
  const std::string &contract) const
{
  auto it = dispatcher_methods.find(contract);
  if (it != dispatcher_methods.end())
    return it->second;

  std::map<std::string, std::vector<std::string>> calls;
  const std::string disp_id =
    "sol:@C@" + contract + "@_ESBMC_Nondet_Extcall_" + contract + "#";
  if (const symbolt *disp = ns.lookup(irep_idt(disp_id)))
    collect_dispatch_calls(disp->value, "sol:@C@" + contract + "@F@", calls);

  return dispatcher_methods.emplace(contract, std::move(calls)).first->second;
}

// Whether Solidity forbids `new` on this contract (abstract / interface /
// library). The frontend stamps `#sol_no_new` on the contract's constructor
// symbol (`sol:@C@<C>@F@<C>#<id>`); scan for it, mirroring get_method_params.
static bool
contract_is_non_instantiable(const namespacet &ns, const std::string &contract)
{
  const std::string ctor_prefix = "sol:@C@" + contract + "@F@" + contract + "#";
  bool no_new = false;
  ns.get_context().foreach_operand([&](const symbolt &s) {
    if (no_new || !s.type.is_code())
      return;
    if (
      has_prefix(s.id.as_string(), ctor_prefix) &&
      s.type.get_bool("#sol_no_new"))
      no_new = true;
  });
  return no_new;
}

// Whether `contract` is a Solidity library. The frontend stamps `#sol_library`
// on the library's constructor symbol (`sol:@C@<C>@F@<C>#<id>`, alongside
// `#sol_no_new`); scan for it, mirroring contract_is_non_instantiable. A
// library is called statically (`Lib.fn(args)`, no instance), which sets it
// apart from an abstract contract / interface (equally non-instantiable but
// not statically callable).
static bool
contract_is_library(const namespacet &ns, const std::string &contract)
{
  const std::string ctor_prefix = "sol:@C@" + contract + "@F@" + contract + "#";
  bool is_lib = false;
  ns.get_context().foreach_operand([&](const symbolt &s) {
    if (is_lib || !s.type.is_code())
      return;
    if (
      has_prefix(s.id.as_string(), ctor_prefix) &&
      s.type.get_bool("#sol_library"))
      is_lib = true;
  });
  return is_lib;
}

// Whether `contract` is a true Solidity interface (frontend stamps
// `#sol_interface` on its constructor symbol, alongside `#sol_no_new`). Only an
// interface is mocked — it is guaranteed to have no constructor arguments and no
// abstract receive/fallback, so `contract ESBMCMock_<I> is <I> { <stubs> }` is
// always fully implementable. An abstract contract (equally non-instantiable but
// possibly carrying ctor args / an abstract receive-fallback) is NOT mocked.
static bool
contract_is_interface(const namespacet &ns, const std::string &contract)
{
  const std::string ctor_prefix = "sol:@C@" + contract + "@F@" + contract + "#";
  bool is_iface = false;
  ns.get_context().foreach_operand([&](const symbolt &s) {
    if (is_iface || !s.type.is_code())
      return;
    if (
      has_prefix(s.id.as_string(), ctor_prefix) &&
      s.type.get_bool("#sol_interface"))
      is_iface = true;
  });
  return is_iface;
}

// The linearized base contracts of `contract` (excluding self), read from the
// `#sol_bases` list the frontend stamps on the constructor symbol. Empty when
// the contract has no bases. Used to instantiate only the most-derived
// contract — a base is constructed transitively by `new Derived(...)`.
static std::set<std::string>
contract_bases(const namespacet &ns, const std::string &contract)
{
  std::set<std::string> bases;
  const std::string ctor_prefix = "sol:@C@" + contract + "@F@" + contract + "#";
  ns.get_context().foreach_operand([&](const symbolt &s) {
    if (!s.type.is_code() || !has_prefix(s.id.as_string(), ctor_prefix))
      return;
    std::istringstream iss(s.type.get("#sol_bases").as_string());
    std::string b;
    while (iss >> b)
      bases.insert(b);
  });
  return bases;
}

// The interface/contract name of a "CONTRACT:<I>" sol-type, else "".
static std::string contract_iface_of(const std::string &sol_type)
{
  return has_prefix(sol_type, "CONTRACT:") ? sol_type.substr(9) : std::string();
}

const foundry_generator::mock_spec &foundry_generator::build_mock_spec(
  const namespacet &ns,
  const std::string &iface) const
{
  auto cached = mock_specs.find(iface);
  if (cached != mock_specs.end())
    return cached->second;

  mock_spec ms;
  ms.name = iface;

  // Name-collision guard: if the source already declares a contract literally
  // named `ESBMCMock_<iface>`, emitting our mock would redeclare it. Degrade to
  // UNSUPPORTED rather than emit an uncompilable duplicate declaration.
  {
    const std::string clash_ctor =
      "sol:@C@ESBMCMock_" + iface + "@F@ESBMCMock_" + iface + "#";
    bool clash = false;
    ns.get_context().foreach_operand([&](const symbolt &s) {
      if (!clash && has_prefix(s.id.as_string(), clash_ctor))
        clash = true;
    });
    if (clash)
      return mock_specs.emplace(iface, std::move(ms)).first->second;
  }

  // Enumerate the interface's externally-visible functions. Inherited methods
  // are already inlined into the derived interface's `@F@` set (verified: a
  // `IChild is IBase` carries `IChild@F@<base-method>`), so no base walk is
  // needed. A genuine method is a CODE symbol whose first argument is the `this`
  // self-pointer; events (`void(a,b,c)`, no `this`) are thereby excluded. The
  // only synthesized code-with-`this` members an interface carries are the
  // `$`-prefixed call helpers (`$call`/`$send`/…) and its own constructor
  // (name == iface); everything else is a real interface method (including a
  // legally `_`-prefixed one), so those two are the only exclusions.
  const std::string fpfx = "sol:@C@" + iface + "@F@";
  std::vector<std::pair<const symbolt *, std::string>> fns; // (symbol, name)
  std::set<std::string> seen_sig; // dedup by (name + param-type-list)
  ns.get_context().foreach_operand([&](const symbolt &s) {
    const std::string id = s.id.as_string();
    if (!s.type.is_code() || !has_prefix(id, fpfx))
      return;
    // `<fpfx><name>#<node>`: reject params/locals (`<fpfx><name>@..`).
    const std::string rest = id.substr(fpfx.size());
    size_t h = rest.find('#');
    if (h == std::string::npos || rest.find('@') != std::string::npos)
      return;
    const std::string name = rest.substr(0, h);
    if (name.empty() || name == iface || name[0] == '$')
      return;
    const code_typet &ct = to_code_type(s.type);
    if (
      ct.arguments().empty() ||
      ct.arguments().front().get_base_name().as_string() != "this")
      return; // an event / free function, not an interface method
    fns.emplace_back(&s, name);
  });

  if (fns.empty())
    return mock_specs.emplace(iface, std::move(ms)).first->second;

  // A stub must NAME a `string` param/return type — but `effective_sol_type`
  // deliberately does not surface STRING (a string call-argument stays
  // UNSUPPORTED rather than default to "" and risk a wrong-branch replay). So
  // recognize the string pointer locally, for the mock stub signature only.
  auto mock_string_type = [](const typet &t) -> std::string {
    return (t.is_pointer() && t.get("#sol_type").as_string() == "STRING")
             ? std::string("STRING")
             : std::string();
  };

  for (const auto &fnp : fns)
  {
    const symbolt *fn = fnp.first;
    const std::string &name = fnp.second;
    const code_typet &ct = to_code_type(fn->type);

    // Parameter types (skip the `this` self-pointer). A parameter type that
    // cannot be named as a Solidity type makes the whole interface unrenderable.
    std::string params;
    std::string sig = name + "(";
    bool ok = true;
    for (size_t i = 1; i < ct.arguments().size(); ++i)
    {
      std::string st = arg_sol_type(ct.arguments()[i]);
      if (st.empty())
        st = mock_string_type(ct.arguments()[i].type());
      const std::string sty = sol_type_to_solidity(st);
      if (sty.empty())
      {
        ok = false;
        break;
      }
      // A reference type in an external signature needs a data location.
      std::string loc;
      if (st == "STRING" || st == "BYTES_DYN" || has_prefix(st, "ARRAY:"))
        loc = " memory";
      params += (params.empty() ? "" : ", ") + sty + loc;
      sig += st + ",";
    }
    if (!ok)
      return mock_specs.emplace(iface, std::move(ms)).first->second;
    if (!seen_sig.insert(sig).second)
      continue; // duplicate signature (already emitted) — skip

    // Return type. Only a single, nameable return is supported; a tuple / struct
    // / unnameable return degrades the whole interface (all-or-nothing).
    std::string ret_sol, ret_default, ret_loc;
    const typet &rt = ct.return_type();
    if (rt.id() != "empty" && !rt.is_nil())
    {
      std::string rst = effective_sol_type(rt);
      if (rst.empty())
        rst = mock_string_type(rt);
      ret_sol = sol_type_to_solidity(rst);
      ret_default = default_sol_literal(rst);
      if (ret_sol.empty() || ret_default.empty())
        return mock_specs.emplace(iface, std::move(ms)).first->second;
      if (rst == "STRING" || rst == "BYTES_DYN" || has_prefix(rst, "ARRAY:"))
        ret_loc = " memory";
    }

    // Mutability: mirror payable (a `pure` override of a payable interface fn is
    // rejected by solc); every other mutability is validly tightened to `pure`.
    const std::string mut =
      fn->type.get_bool("#sol_payable") ? "payable" : "pure";

    std::string stub =
      "  function " + name + "(" + params + ") external " + mut + " override";
    if (!ret_sol.empty())
      stub +=
        " returns (" + ret_sol + ret_loc + ") { return " + ret_default + "; }";
    else
      stub += " {}";
    ms.stubs.push_back(stub);
  }

  ms.renderable = !ms.stubs.empty();
  return mock_specs.emplace(iface, std::move(ms)).first->second;
}

// Model value of a focused-function parameter (contract,method,param), read
// directly from the solver: locate any symbol subexpr with that base name in a
// taken (guard-true) SSA step and query its model value. In `--function` mode
// the entry calls the target with nil arguments, so parameters are free nondet
// inputs with no `param = nondet` assignment to key off (unlike the dispatcher
// path). Returns a null expr when the parameter is absent (sliced away because
// it is irrelevant to the covered branch — the caller then uses the type
// default, which is sound).
// Reconstruct the length of a nondet `string` argument from the model. The
// harness fills the fixed global buffer `_ESBMC_rand_str` with `len` non-null
// bytes and zeroes the tail (nondet_string, solidity_string.c), so the leading
// non-null run of any guard-true model value of that buffer is the string
// length the covered path used. Returns the maximum such run over all
// guard-true SSA steps (0 when the string is empty / unread on this path). The
// buffer is shared across all nondet strings in a run, so this is a single
// per-counterexample length — sufficient for the single-string harness entries
// these tests exercise.
static unsigned recover_nondet_string_length(
  const symex_target_equationt &target,
  smt_convt &smt_conv)
{
  unsigned best = 0;
  auto leading_nonzero = [](const expr2tc &arr) -> unsigned {
    if (!arr || !is_constant_array2t(arr))
      return 0;
    const auto &m = to_constant_array2t(arr).datatype_members;
    unsigned n = 0;
    for (const auto &e : m)
    {
      if (!e || !is_constant_int2t(e) || to_constant_int2t(e).value == 0)
        break;
      ++n;
    }
    return n;
  };
  std::function<void(const expr2tc &)> visit = [&](const expr2tc &e) {
    if (!e)
      return;
    if (
      is_symbol2t(e) &&
      to_symbol2t(e).thename.as_string().find("rand_str") != std::string::npos)
    {
      unsigned n = leading_nonzero(smt_conv.get(e));
      if (n > best)
        best = n;
    }
    e->foreach_operand([&](const expr2tc &s) { visit(s); });
  };
  for (auto const &step : target.SSA_steps)
  {
    if (!smt_conv.l_get(step.guard_ast).is_true())
      continue;
    if (step.is_assignment())
    {
      visit(step.lhs);
      visit(step.rhs);
    }
    if (step.is_assume() || step.is_assert())
      visit(step.cond);
  }
  return best;
}

expr2tc foundry_generator::recover_focus_param(
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  const std::string &contract,
  const std::string &method,
  const std::string &param)
{
  expr2tc found;
  std::function<void(const expr2tc &)> visit = [&](const expr2tc &e) {
    if (!e || found)
      return;
    if (is_symbol2t(e))
    {
      std::string c, m, p;
      if (
        foundry_generator::parse_param_symbol(
          to_symbol2t(e).thename.as_string(), c, m, p) &&
        c == contract && m == method && p == param)
      {
        found = e;
        return;
      }
    }
    e->foreach_operand([&](const expr2tc &sub) { visit(sub); });
  };
  for (auto const &step : target.SSA_steps)
  {
    if (found)
      break;
    if (!smt_conv.l_get(step.guard_ast).is_true())
      continue;
    if (step.is_assignment())
      visit(step.rhs);
    if (step.is_assume() || step.is_assert())
      visit(step.cond);
  }
  return found ? smt_conv.get(found) : expr2tc();
}

// Declared parameters (source order, `#sol_type`) of an exact method symbol id,
// skipping the synthesised `this` self-pointer. Used to resolve overloads,
// where the base name alone is ambiguous.
static std::vector<std::pair<std::string, std::string>>
params_of_method_id(const namespacet &ns, const std::string &id)
{
  std::vector<std::pair<std::string, std::string>> params;
  const symbolt *fn = ns.lookup(irep_idt(id));
  if (!fn || !fn->type.is_code())
    return params;
  for (const auto &arg : to_code_type(fn->type).arguments())
  {
    std::string pname = arg.get_base_name().as_string();
    if (pname.empty() || pname == "this")
      continue;
    // Prefer the width stamped on the argument (arg_sol_type), then the
    // parameter symbol's type when the argument carries no source tag.
    std::string st = arg_sol_type(arg);
    if (st.empty())
    {
      const irep_idt &pid = arg.get_identifier();
      if (const symbolt *ps = !pid.empty() ? ns.lookup(pid) : nullptr)
        st = effective_sol_type(ps->type);
    }
    params.emplace_back(pname, st);
  }
  return params;
}

// Whether a resolved method id is a `payable` function (carries the
// `#sol_payable` flag stamped by get_function_definition). Only payable methods
// may receive `{value: N}` — sending value to a non-payable method reverts.
static bool symbol_is_payable(const namespacet &ns, const std::string &id)
{
  if (id.empty())
    return false;
  const symbolt *fn = ns.lookup(irep_idt(id));
  return fn && fn->type.is_code() && fn->type.get_bool("#sol_payable");
}

// Wrap an integer literal in an explicit typed cast so Solidity's overload
// resolution selects the intended overload; bool/address literals are already
// unambiguous and pass through.
static std::string
cast_for_overload(const std::string &sol_type, const std::string &lit)
{
  std::string ty;
  if (has_prefix(sol_type, "UINT"))
    ty = "uint" + sol_type.substr(4);
  else if (has_prefix(sol_type, "INT"))
    ty = "int" + sol_type.substr(3);
  else
    return lit;
  if (ty == "uint")
    ty = "uint256";
  if (ty == "int")
    ty = "int256";
  return ty + "(" + lit + ")";
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

// If `raw` is a dispatcher's FIRST nondet guard
// (`sol:@C@<C>@_ESBMC_Nondet_Extcall_<C>#::$tmp::return_value$_nondet_bool$1`),
// return <C>; else "". Each such guard marks the start of one transaction: the
// harness re-enters the dispatcher loop and re-evaluates guard #1.
static std::string dispatcher_tx_contract(const std::string &raw)
{
  const std::string mark = "_ESBMC_Nondet_Extcall_";
  size_t p = raw.find(mark);
  if (p == std::string::npos || p < 8 || !has_prefix(raw, "sol:@C@"))
    return "";
  if (raw[p - 1] != '@')
    return "";
  const std::string guard = "return_value$_nondet_bool$";
  size_t gp = raw.find(guard);
  if (gp == std::string::npos)
    return "";
  size_t d = gp + guard.size();
  std::string num;
  while (d < raw.size() && isdigit(static_cast<unsigned char>(raw[d])))
    num += raw[d++];
  if (num != "1")
    return "";
  return raw.substr(7, p - 1 - 7);
}

// Base method name a step is executing, from its source-location function. The
// Solidity frontend stores either the source name ("probe") or the mangled id
// (`sol:@C@<C>@F@<m>#..`); both reduce to the source method name.
static std::string
step_location_method(const symex_target_equationt::SSA_stept &step)
{
  std::string fn = step.source.pc->location.function().as_string();
  if (fn.empty())
    return "";
  if (has_prefix(fn, "sol:@C@"))
  {
    size_t f = fn.find("@F@");
    if (f == std::string::npos)
      return "";
    std::string after = fn.substr(f + 3);
    size_t cut = after.find_first_of("#@");
    return cut == std::string::npos ? after : after.substr(0, cut);
  }
  if (has_prefix(fn, "c::"))
    fn = fn.substr(3);
  return fn;
}

foundry_generator::test_case foundry_generator::reconstruct(
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  const namespacet &ns) const
{
  // Length of a nondet `string` argument recovered from the model (see
  // recover_nondet_string_length); used by build_call to render a string
  // literal that reproduces a `bytes(s).length` branch. A single value per
  // counterexample — the harness shares one nondet string buffer.
  const unsigned recovered_str_len =
    recover_nondet_string_length(target, smt_conv);

  // Resolve a recovered (contract, method, args) into a compilable call by
  // matching the method's DECLARED parameters in source order: a recovered
  // value fills its slot, an un-recovered slot (parameter sliced away because
  // it is irrelevant to the covered branch, or a short-circuited operand) takes
  // a type default — sound, since a sliced argument cannot change which branch
  // is reached — and an unrenderable type flags the call unsupported. This is
  // always arity-correct: never a wrong-arity call.
  auto build_call = [&](
                      const std::string &contract,
                      const std::string &method,
                      const std::map<std::string, sol_arg> &recovered) {
    sol_call out;
    out.contract = contract;
    out.method = method;

    // receive() / fallback() are special functions Solidity forbids calling by
    // name (`c.receive()` does not compile). They are reached in the EVM via a
    // low-level `address(c).call{value:}("")` / with mismatched calldata — not
    // yet reconstructed here — so flag unsupported rather than emit an
    // uncompilable named call. (The frontend normalizes their names to
    // "receive"/"fallback"; a regular function cannot carry these names.)
    if (method == "receive" || method == "fallback")
    {
      out.supported = false;
      return out;
    }

    // Resolve the exact method signature. For a dispatcher-callable method we
    // use its exact id (disambiguating overloads by which recovered parameter
    // names the candidate declares); a constructor or otherwise-unlisted method
    // falls back to base-name lookup.
    const auto &callable = dispatcher_callable(ns, contract);
    auto cit = callable.find(method);
    std::vector<std::pair<std::string, std::string>> decls;
    bool overloaded = false;
    std::string chosen_id; // exact resolved method id, for payability lookup

    if (cit != callable.end() && !cit->second.empty())
    {
      const auto &ids = cit->second;
      if (ids.size() == 1)
      {
        chosen_id = ids.front();
        decls = params_of_method_id(ns, ids.front());
      }
      else
      {
        // Overloaded: pick the single candidate declaring every recovered
        // parameter name. No unique match (incl. no recovered args) -> we
        // cannot know which overload ran, so mark unsupported rather than guess.
        overloaded = true;
        std::string chosen;
        for (const auto &id : ids)
        {
          auto p = params_of_method_id(ns, id);
          bool ok = true;
          for (const auto &kv : recovered)
            if (std::none_of(p.begin(), p.end(), [&](const auto &d) {
                  return d.first == kv.first;
                }))
            {
              ok = false;
              break;
            }
          if (ok && !recovered.empty())
          {
            if (!chosen.empty())
            {
              chosen.clear(); // ambiguous
              break;
            }
            chosen = id;
            chosen_id = id;
            decls = std::move(p);
          }
        }
        if (chosen.empty())
        {
          for (const auto &kv : recovered)
            out.args.push_back(kv.second);
          out.supported = false;
          return out;
        }
      }
    }
    else
    {
      decls = get_method_params(ns, contract, method);
      if (decls.empty() && !recovered.empty())
      {
        // Unknown signature: keep recovered args but flag unsupported so we
        // never emit a possibly-wrong call shape.
        for (const auto &kv : recovered)
          out.args.push_back(kv.second);
        out.supported = false;
        return out;
      }
    }

    for (const auto &decl : decls)
    {
      sol_arg a;
      a.param = decl.first;
      a.sol_type = decl.second;

      // `string` argument: the recovered value is a bare `char *` pointer, so
      // format_sol_value/default cannot derive a length from it. Instead render
      // a Solidity string literal of the length reconstructed from the model
      // (recovered_str_len). This reproduces a `bytes(s).length > k` branch
      // with a genuinely long-enough argument, and its complement with a short
      // / empty one — rather than both arms collapsing onto the empty default
      // and deduplicating to a single non-reaching case. Content is filler
      // ('a'), which is faithful for length-based branches; a content-dependent
      // branch is a documented residual (the try/catch wrap tolerates it).
      if (decl.second == "STRING")
      {
        unsigned slen = recovered_str_len > 256 ? 256 : recovered_str_len;
        a.literal = "\"" + std::string(slen, 'a') + "\"";
        a.defaulted = false; // reconstructed from the model length
        out.args.push_back(a);
        continue;
      }

      // Interface/contract-typed argument: pass a synthesized mock instance
      // rather than a literal (a bare address reverts when the contract calls a
      // method on the handle). Only a true INTERFACE is mocked — it is
      // guaranteed to have no constructor arguments and no abstract
      // receive/fallback, so `ESBMCMock_<I> is <I>` is always fully
      // implementable. An abstract contract (ctor args / abstract
      // receive-fallback) or a concrete contract (real side effects a later
      // branch may depend on) is NOT mocked and degrades to UNSUPPORTED.
      // All-or-nothing: an interface whose full stub set cannot render is
      // UNSUPPORTED.
      //
      // Distinctness: each interface argument gets its OWN mock instance
      // (keyed by parameter name). ESBMC's recovered construction path already
      // satisfied any `a != b` guard, and a fresh instance per slot reproduces
      // it (distinct deployed addresses). A constructor that instead REQUIRES
      // two interface parameters to be the SAME instance (`a == b`) is not
      // reproduced — the concrete `$address` the solver equated is not
      // recoverable from a pointer model value (get() returns an unconstrained
      // symbol for both) — that rare shape would revert setUp; it is a
      // documented limitation, not a silent wrong-coverage claim.
      const std::string iface = contract_iface_of(decl.second);
      if (!iface.empty())
      {
        if (
          contract_is_interface(ns, iface) &&
          build_mock_spec(ns, iface).renderable)
        {
          a.mock_iface = iface;
          a.mock_key = decl.first; // fresh instance per slot
          a.literal = "mk_" + iface + "_" + a.mock_key; // deployed mock var
        }
        else
          out.supported = false; // literal stays empty
        out.args.push_back(a);
        continue;
      }

      // Struct-typed argument: render a positional struct literal from the
      // recovered constant_struct + the declared struct tag (source field
      // types). Detected by the recovered value being a constant_struct (the
      // declared sol_type is "" for a user struct). All-or-nothing inside
      // format_struct_literal.
      {
        auto sit = recovered.find(decl.first);
        // A user struct value is a constant_struct — but so are `bytesN`
        // (BytesStatic) and `bytes` (BytesDynamic) values, which have their own
        // renderers; exclude those by struct tag so they are not mis-routed here.
        bool is_user_struct = false;
        if (
          sit != recovered.end() && sit->second.value &&
          is_constant_struct2t(sit->second.value))
        {
          const type2tc &vt = to_constant_struct2t(sit->second.value).type;
          is_user_struct =
            is_struct_type(vt) &&
            !is_bytes_wrapper_struct(to_struct_type(vt).name.as_string());
        }
        if (is_user_struct)
        {
          std::string qualified;
          a.literal = format_struct_literal(
            ns, sit->second.value, qualified, a.struct_imports);
          if (a.literal.empty())
            out.supported = false;
          else
            a.sol_type = "STRUCT:" + qualified; // for import resolution
          out.args.push_back(a);
          continue;
        }
      }

      auto it = recovered.find(decl.first);
      // Re-format any recovered value against the DECLARED type (decl.second),
      // which carries the authoritative source width — critical for bytesN,
      // whose width must be the declared width, not one inferred from the
      // recovered struct's `.length`. The recovery-site literal was computed
      // from the (possibly width-degraded) parameter symbol type, so only use
      // it as a fallback when re-formatting yields nothing.
      if (it != recovered.end() && it->second.value)
        a.literal = format_sol_value(decl.second, it->second.value);
      if (a.literal.empty() && it != recovered.end())
        a.literal = it->second.literal;
      const bool from_recovered = !a.literal.empty();
      if (a.literal.empty())
        a.literal = default_sol_literal(decl.second);
      // A non-empty literal not sourced from a recovered value is a type default.
      a.defaulted = !a.literal.empty() && !from_recovered;
      if (a.literal.empty())
        out.supported = false;
      else if (overloaded)
        a.literal = cast_for_overload(decl.second, a.literal);
      out.args.push_back(a);
    }
    out.payable = symbol_is_payable(ns, chosen_id);
    return out;
  };

  // Constructor arguments (assigned before the dispatcher loop) and one segment
  // per transaction. The harness constructs every contract up front, then loops
  // dispatcher invocations, each running at most one public method chosen by
  // its first nondet guard. Split at those guards: before the first is
  // construction; each later segment is one transaction whose method is the
  // first contract-method body that runs in it.
  std::map<std::string, std::map<std::string, sol_arg>> ctor_args;
  // ③A0 constructor-time env. A ctor branching on block.timestamp is satisfied
  // by the deploy-time ambient (set in initialize(), before the dispatcher loop);
  // Foundry's `new C()` deploys under the default timestamp, so the ctor can
  // revert in setUp and fail the whole suite. Recover the deploy-time value and
  // whether the ctor reads it, to emit `vm.warp` before the deploy.
  expr2tc ctor_block_timestamp;
  bool ctor_reads_timestamp = false;
  // ③A0 constructor-time msg.sender (deployer identity). A ctor binding
  // `owner = msg.sender` stores the deploy-time sender; recover it so setUp can
  // `vm.startPrank` before `new C()`, making a later `onlyOwner` check match (or
  // mismatch) a known owner. `ctor_sender_dirty` guards against a ctor whose own
  // body makes a nested call that overwrites msg_sender before the owner bind.
  expr2tc ctor_msg_sender;
  bool ctor_reads_msg_sender = false;
  bool ctor_sender_shadowed = false;
  bool ctor_sender_dirty = false;
  // The contract setUp actually deploys is the run's --contract; attribute the
  // ctor warp there rather than to step_location_method (which for a base ctor,
  // an inlined modifier wrapper, or a user init helper names the wrong thing —
  // e.g. an aux `C_afterStart` that is not a real contract).
  const std::string ctor_ts_contract = config.options.get_option("contract");
  // ③A0 constructor-time msg.value. A `payable` ctor that requires/branches on
  // msg.value must be deployed with `new C{value: v}(...)` (and the test funded
  // via vm.deal), else a `require(msg.value >= N)` reverts setUp and fails the
  // whole suite. Recover the deploy-time value + whether the ctor reads it; only
  // emit for a payable ctor (sending value to a non-payable ctor cannot compile).
  expr2tc ctor_msg_value;
  bool ctor_reads_value = false;
  bool ctor_is_payable = false;
  {
    // The constructor symbol is `sol:@C@<C>@F@<C>#<id>`; read its payability.
    const std::string ctor_pfx =
      "sol:@C@" + ctor_ts_contract + "@F@" + ctor_ts_contract + "#";
    ns.get_context().foreach_operand([&](const symbolt &s) {
      if (
        s.type.is_code() && has_prefix(s.id.as_string(), ctor_pfx) &&
        s.type.get_bool("#sol_payable"))
        ctor_is_payable = true;
    });
  }
  struct segment
  {
    std::string contract, method;
    std::map<std::string, sol_arg> args;
    bool reverts = false;
    expr2tc msg_value;       // ③A0: solver-picked msg.value for this tx
    expr2tc block_timestamp; // ③A0: solver-picked block.timestamp for this tx
    bool reads_timestamp = false; // ③A0: this tx's body reads block.timestamp
    expr2tc msg_sender; // ③A0: solver-picked top-level msg.sender for this tx
    bool reads_msg_sender = false; // ③A0: this tx's body reads msg.sender
    // ③A0: msg_sender currently holds a nested-call sender (a wrapper installed
    // the callee's identity and has not restored it yet), so a read taken here
    // did NOT see the top-level value. Tracked by model value, not by "a write
    // occurred" — see the write handler for why the syntactic test is wrong.
    bool sender_shadowed = false;
    // ③A0: a read of msg.sender happened while shadowed, so the top-level
    // `vm.prank` value is NOT what that branch read. When dirty we refuse to
    // pin the sender (safe under-coverage) rather than emit a call whose
    // msg.sender the replay cannot faithfully reproduce.
    bool sender_dirty = false;
  };
  std::vector<segment> segs;

  // ③A0 environment pinning. `_sol_per_tx_reseed` assigns the globals
  // `msg_value` / `block_timestamp` in the dispatcher prologue JUST BEFORE the
  // tx's Extcall guard marker, so the reseed for tx N precedes segment N's push.
  // Buffer the most recent recovered values and attach them to the next segment
  // when its marker fires.
  expr2tc pending_msg_value, pending_block_timestamp, pending_msg_sender;
  // True iff the symbol NAME denotes the EVM ambient C global `base`
  // (solidity_blockchain.c), NOT a user Solidity symbol that merely ends with
  // the same word. User symbols live in the `sol:` namespace
  // (`sol:@C@<C>@F@<f>@block_timestamp`); the C globals do not, so a `sol:`
  // prefix is a decisive exclusion — otherwise a user variable named
  // `block_timestamp` / `msg_value` would false-match and trigger a spurious pin.
  auto is_env_global = [](const std::string &name, const char *base) -> bool {
    if (has_prefix(name, "sol:"))
      return false;
    const size_t at = name.rfind('@');
    return (at == std::string::npos ? name : name.substr(at + 1)) == base;
  };
  auto lhs_is_env_global = [&](const expr2tc &lhs, const char *base) -> bool {
    return lhs && is_symbol2t(lhs) &&
           is_env_global(to_symbol2t(lhs).thename.as_string(), base);
  };
  // True if `e` references the EVM global `base` (used to detect whether a
  // segment body actually reads block.timestamp).
  std::function<bool(const expr2tc &, const char *)> reads_global =
    [&](const expr2tc &e, const char *base) -> bool {
    if (!e)
      return false;
    if (is_symbol2t(e))
      return is_env_global(to_symbol2t(e).thename.as_string(), base);
    bool found = false;
    e->foreach_operand([&](const expr2tc &sub) {
      if (!found && reads_global(sub, base))
        found = true;
    });
    return found;
  };
  // Whether an SSA step belongs to a helper we must exclude from body-level
  // read detection (the reseed's own monotonic block.timestamp read, and the
  // one-time initialize()), so it is not mis-attributed to the contract body.
  auto step_in_sol = [](const symex_target_equationt::SSA_stept &step) -> bool {
    const std::string f = step.source.pc->location.file().as_string();
    return f.size() >= 4 && f.compare(f.size() - 4, 4, ".sol") == 0;
  };
  // A step is an ambient/library helper (excluded from user-level block.timestamp
  // read detection) iff it is NOT in a user `.sol` source. All user
  // contract/ctor/modifier/init code lives in a `.sol` file; the ambient model
  // (`initialize` / `_sol_per_tx_reseed` and their internal block.timestamp
  // reads) lives in the C library. `location.function()` returns the bare short
  // name for BOTH a user `initialize()` and the library one, so the file — not
  // the function name — is the reliable discriminator.
  auto is_env_helper_step =
    [&](const symex_target_equationt::SSA_stept &step) -> bool {
    return !step_in_sol(step);
  };

  // Resolve a covered function name to the dispatcher-callable method that owns
  // it. A method with modifiers runs its body inside a synthetic wrapper
  // `<method>_<modifier>` (get_modifier_function_name) that the dispatcher never
  // calls directly, so a branch covered inside the wrapper names the wrapper,
  // not the external entry — and the segment would get no method (dropping the
  // call, or falling back without env pins). The wrapper symbol carries
  // `#sol_modifier_wrapper_for` (stamped by the frontend) naming its real
  // method; that is authoritative, unlike a name-prefix guess which is
  // ambiguous because `_` is a legal identifier char (`a_b_mod` could be
  // `a`+`b_mod` or `a_b`+`mod`). Returns the callable method, or "" when `name`
  // is neither callable nor a wrapper of a callable method.
  auto resolve_dispatcher_method =
    [&](const std::string &contract, const std::string &name) -> std::string {
    const auto &callable = dispatcher_callable(ns, contract);
    if (callable.count(name))
      return name;
    if (
      const symbolt *s =
        ns.lookup(irep_idt("sol:@C@" + contract + "@F@" + name + "#0")))
    {
      const std::string real =
        s->type.get("#sol_modifier_wrapper_for").as_string();
      if (!real.empty() && callable.count(real))
      {
        // Guard against the frontend's unescaped `<method>_<modifier>` wrapper
        // naming: `a`+modifier `b_mod` and `a_b`+modifier `mod` both produce
        // wrapper `a_b_mod`, collapsing to one symbol whose single stamp may
        // name the wrong sibling. Resolve ONLY when exactly one
        // dispatcher-callable method is a `<m>_` prefix of the wrapper name —
        // i.e. the name is unambiguous. Otherwise under-cover (return "")
        // rather than risk emitting a call to the wrong real method.
        unsigned producers = 0;
        for (const auto &kv : callable)
        {
          const std::string &m = kv.first;
          if (
            name.size() > m.size() + 1 && name.compare(0, m.size(), m) == 0 &&
            name[m.size()] == '_')
            ++producers;
        }
        if (producers == 1)
          return real;
      }
    }
    return "";
  };

  for (auto const &step : target.SSA_steps)
  {
    if (!smt_conv.l_get(step.guard_ast).is_true())
      continue;

    // Revert fidelity: the covered branch-coverage claim is the one kept assert
    // whose guard is true on this path. When goto_coverage marked its edge as a
    // DETECTED revert (Phase A: a `revert CustomError(...)` reaching a
    // `#sol_error` call — NOT `require(cond, ...)` failures, which lower with no
    // detectable terminator), the transaction active at this step reverts, so
    // its call is wrapped in a precise vm.expectRevert(). Undetected reverts are
    // caught by the try/catch fallback in write_foundry_file instead.
    if (
      step.is_assert() && !segs.empty() &&
      step.source.pc->location.get_bool("sol_revert_edge"))
      segs.back().reverts = true;

    // The covered coverage claim (a guard-true assert) is AUTHORITATIVE for the
    // active transaction's method: its source location names the function whose
    // branch is covered. In a multi-function whole-unit dispatcher the segment's
    // method is otherwise guessed from the first recovered param / executed body
    // and can latch onto the WRONG function (e.g. a dock claim mis-attributed to
    // ship), so override the guess with the covered method here.
    if (step.is_assert() && !segs.empty())
    {
      const std::string cm = resolve_dispatcher_method(
        segs.back().contract, step_location_method(step));
      if (!cm.empty())
        segs.back().method = cm;
    }

    const bool assign_sym =
      step.is_assignment() && is_symbol2t(step.original_lhs);
    const std::string lhs_id =
      assign_sym ? to_symbol2t(step.original_lhs).thename.as_string()
                 : std::string();

    // ③A0: recover the per-tx msg.value from the reseed assignment. The RHS is
    // a bare NONDET, so read the model value off the renamed LHS. Buffered as
    // pending; attached to the segment pushed next. Accept ONLY the assignment
    // in `_sol_per_tx_reseed` (the dispatcher prologue), NOT the transient
    // `msg_value` writes that value-forwarding call wrappers emit to set an
    // inner call's context — those would mis-attribute an inner-call value to
    // the next top-level transaction.
    const bool from_reseed =
      step.source.pc->location.function().as_string().find(
        "_sol_per_tx_reseed") != std::string::npos;
    if (
      assign_sym && from_reseed &&
      lhs_is_env_global(step.original_lhs, "msg_value"))
    {
      pending_msg_value = smt_conv.get(step.lhs);
      continue;
    }
    // ③A0: recover per-tx block.timestamp from the reseed (same discipline).
    if (
      assign_sym && from_reseed &&
      lhs_is_env_global(step.original_lhs, "block_timestamp"))
    {
      pending_block_timestamp = smt_conv.get(step.lhs);
      continue;
    }
    // ③A0 ctor-time: capture the deploy-time block.timestamp (set in
    // initialize() before the dispatcher loop) so a ctor reading it can be
    // deployed under a matching vm.warp in setUp.
    // The deploy-time block.timestamp is set by the library `initialize()` (in
    // the C model, NOT a `.sol` file); capture it so a ctor reading
    // block.timestamp can be deployed under a matching vm.warp.
    const std::string step_fn = step.source.pc->location.function().as_string();
    if (
      assign_sym && segs.empty() &&
      lhs_is_env_global(step.original_lhs, "block_timestamp") &&
      !step_in_sol(step) && step_fn.find("initialize") != std::string::npos)
    {
      ctor_block_timestamp = smt_conv.get(step.lhs);
      continue;
    }
    // ③A0 ctor-time: capture the deploy-time msg.value (set in initialize())
    // so a payable ctor reading it can be deployed with a matching {value:}.
    if (
      assign_sym && segs.empty() &&
      lhs_is_env_global(step.original_lhs, "msg_value") && !step_in_sol(step) &&
      step_fn.find("initialize") != std::string::npos)
    {
      ctor_msg_value = smt_conv.get(step.lhs);
      continue;
    }
    // ③A0: recover msg.sender. Three msg_sender writes exist: the per-tx reseed
    // (top-level sender for the next segment), the deploy-time initialize() (the
    // deployer identity a ctor stores as `owner`), and nested/high/low-level
    // call wrappers that overwrite it with the caller address. The last makes
    // the active tx's top-level sender un-reproducible via `vm.prank`, so it
    // marks the segment (or ctor) sender-dirty and no sender pin is emitted.
    if (assign_sym && lhs_is_env_global(step.original_lhs, "msg_sender"))
    {
      if (from_reseed)
      {
        pending_msg_sender = smt_conv.get(step.lhs);
        continue;
      }
      if (
        segs.empty() && !step_in_sol(step) &&
        step_fn.find("initialize") != std::string::npos)
      {
        ctor_msg_sender = smt_conv.get(step.lhs);
        continue;
      }
      // Any other write is a call wrapper installing the callee's sender (and,
      // paired with it, the restore on the way out). Whether it actually
      // shadows the top-level sender cannot be decided syntactically: symex
      // merges a branch-local assignment into an UNCONDITIONAL step whose RHS
      // is `cond ? new : old`, so a wrapper this path never entered still
      // appears here with a true guard. Treating "was written" as "was
      // overwritten" therefore poisons EVERY path of any contract that
      // contains a `.transfer()` / `.call()` anywhere, even one covering a
      // method that makes no call at all. Decide on the MODEL value instead:
      // equal to the transaction's top-level sender means this write left the
      // sender unchanged on this path (an untaken-branch merge, or the
      // wrapper's own restore), so a top-level `vm.prank` still reproduces it.
      const expr2tc &top_sender =
        segs.empty() ? ctor_msg_sender : segs.back().msg_sender;
      const expr2tc now_sender = smt_conv.get(step.lhs);
      const bool shadowed =
        !top_sender || !now_sender || now_sender != top_sender;
      if (segs.empty())
        ctor_sender_shadowed = shadowed;
      else
        segs.back().sender_shadowed = shadowed;
      continue;
    }

    // Transaction boundary: a fresh dispatcher invocation.
    if (assign_sym)
    {
      std::string txc = dispatcher_tx_contract(lhs_id);
      if (!txc.empty())
      {
        segment s;
        s.contract = txc;
        s.msg_value = pending_msg_value;
        s.block_timestamp = pending_block_timestamp;
        s.msg_sender = pending_msg_sender;
        segs.push_back(std::move(s));
        pending_msg_value = expr2tc();
        pending_block_timestamp = expr2tc();
        pending_msg_sender = expr2tc();
        continue;
      }
    }

    // ③A0: does this tx's BODY read block.timestamp? Only then is a vm.warp
    // faithful and non-noisy. Exclude env-helper steps (reseed/initialize) so
    // their internal block.timestamp reads are not mis-attributed to the body.
    if (!is_env_helper_step(step))
    {
      const bool a = step.is_assignment();
      const bool c = step.is_assume() || step.is_assert();
      auto reads = [&](const char *g) {
        return (a && reads_global(step.rhs, g)) ||
               (c && reads_global(step.cond, g));
      };
      if (!segs.empty())
      {
        if (!segs.back().reads_timestamp && reads("block_timestamp"))
          segs.back().reads_timestamp = true;
        // A read taken while a call wrapper holds the sender saw the callee's
        // identity, which `vm.prank` (a TOP-LEVEL pin) cannot reproduce — that,
        // not the bare existence of a wrapper on some other branch, is what
        // makes the pin unfaithful.
        if (reads("msg_sender"))
        {
          if (segs.back().sender_shadowed)
            segs.back().sender_dirty = true;
          else
            segs.back().reads_msg_sender = true;
        }
      }
      // A pre-segment (ctor-body) read means the constructor depends on the
      // deploy-time ambient (attributed to the deploy contract above).
      else
      {
        if (!ctor_reads_timestamp && reads("block_timestamp"))
          ctor_reads_timestamp = true;
        if (reads("msg_sender"))
        {
          if (ctor_sender_shadowed)
            ctor_sender_dirty = true;
          else
            ctor_reads_msg_sender = true;
        }
        if (!ctor_reads_value && reads("msg_value"))
          ctor_reads_value = true;
      }
    }

    // A recovered nondet parameter value.
    expr2tc nondet =
      assign_sym ? symex_slicet::get_nondet_symbol(step.rhs) : expr2tc();
    std::string c, m, p;
    if (nondet && is_symbol2t(nondet) && parse_param_symbol(lhs_id, c, m, p))
    {
      sol_arg a;
      a.param = p;
      if (const symbolt *ps = ns.lookup(irep_idt(lhs_id)))
        a.sol_type = effective_sol_type(ps->type);
      a.value = smt_conv.get(nondet);
      a.literal = format_sol_value(a.sol_type, a.value);

      if (segs.empty())
      {
        if (m == c) // constructor argument (method == contract name)
          ctor_args[c][p] = a;
      }
      else if (c == segs.back().contract)
      {
        segs.back().args[p] = a;
        // Only accept `m` as the transaction's method if the dispatcher can
        // actually call it. A recovered parameter may belong to a modifier/aux
        // helper (`<method>_<modifier>`) the dispatcher never enters directly;
        // its value still belongs to the real method, but its name must not be
        // emitted as the call. The real entry is set from source-location below.
        if (segs.back().method.empty())
        {
          const std::string rm = resolve_dispatcher_method(c, m);
          if (!rm.empty())
            segs.back().method = rm;
        }
      }
      continue;
    }

    // Otherwise fix the transaction's method from the first dispatcher-callable
    // body that executes in it (catches parameterless / sliced-argument calls,
    // and skips modifier/aux helpers that are not external entries).
    if (!segs.empty() && segs.back().method.empty())
    {
      const std::string base = resolve_dispatcher_method(
        segs.back().contract, step_location_method(step));
      if (!base.empty())
        segs.back().method = base;
    }
  }

  test_case calls;

  // Constructor args forwarded to a BASE ctor are recovered under the base
  // contract, not the deploy contract. E.g. EscrowDst's empty body forwards its
  // args to `BaseEscrow(rescueDelay, accessToken)`, so they land in
  // ctor_args[BaseEscrow] while the deploy is `new EscrowDst(...)`. Remap a
  // base's recovered ctor args onto the deploy contract (ctor_ts_contract) by
  // parameter name — build_call fills the deploy ctor's declared params from
  // them. (A base ctor is abstract, so its own entry would be base-dropped
  // anyway; move it so the deploy contract's ctor is the one reconstructed.)
  bool deploy_remapped = false;
  if (!ctor_ts_contract.empty() && !ctor_args.count(ctor_ts_contract))
  {
    for (const auto &base : contract_bases(ns, ctor_ts_contract))
    {
      auto bit = ctor_args.find(base);
      if (bit == ctor_args.end())
        continue;
      for (const auto &kv : bit->second)
        ctor_args[ctor_ts_contract].emplace(kv.first, kv.second);
      ctor_args.erase(bit);
      deploy_remapped = true;
    }
  }

  // --function isolated-function mode: the run focuses one function with no
  // dispatcher, so no transaction segments were found. Reconstruct a single
  // call to the focused function with its recovered parameters. This is the
  // path a Solidity `library` takes (a library has no contract dispatcher);
  // it also serves any contract free-function run in isolation.
  const std::string focus_fn = config.options.get_option("function");
  if (segs.empty() && ctor_args.empty() && !focus_fn.empty())
  {
    // Every contract/library declaring a function with this base name — the
    // owner is disambiguated below by which one renders a supported call.
    std::set<std::string> owners;
    const std::string needle = "@F@" + focus_fn + "#";
    ns.get_context().foreach_operand([&](const symbolt &s) {
      if (!s.type.is_code())
        return;
      const std::string id = s.id.as_string();
      if (!has_prefix(id, "sol:@C@") || id.find(needle) == std::string::npos)
        return;
      size_t f = id.find("@F@");
      if (f != std::string::npos)
        owners.insert(id.substr(7, f - 7)); // <C> between "sol:@C@" and "@F@"
    });
    for (const auto &c : owners)
    {
      // Sound ONLY for a library. A library's internal functions inline into
      // the caller, so the generated test can call `Lib.fn(args)` statically —
      // whereas a contract's private/internal function is not externally
      // callable (emitting `inst.fn(...)` would not compile), and a
      // public/external contract method is reconstructed via the dispatcher
      // path, not here. Restricting to libraries also disambiguates a function
      // name shared across contracts to the one this run actually verified,
      // rather than guessing by "first that renders".
      if (!contract_is_library(ns, c))
        continue;
      std::map<std::string, sol_arg> recovered;
      for (const auto &decl : get_method_params(ns, c, focus_fn))
      {
        expr2tc v =
          recover_focus_param(target, smt_conv, c, focus_fn, decl.first);
        if (!v)
          continue;
        sol_arg a;
        a.param = decl.first;
        a.sol_type = decl.second;
        a.value = v;
        a.literal = format_sol_value(decl.second, v);
        recovered[decl.first] = a;
      }
      sol_call call = build_call(c, focus_fn, recovered);
      if (call.supported)
      {
        calls.push_back(std::move(call));
        break;
      }
    }
  }

  // The deployer identity is pinnable only when the ctor reads msg.sender on a
  // clean (no nested-call overwrite) path — else `vm.startPrank` cannot
  // faithfully reproduce the stored owner.
  const bool ctor_needs_deployer = ctor_reads_msg_sender && !ctor_sender_dirty;
  const std::string ctor_value_lit =
    ctor_msg_value ? format_sol_value("UINT256", ctor_msg_value)
                   : std::string();
  const bool ctor_value_nonzero =
    !ctor_value_lit.empty() && ctor_value_lit != "0";
  // A payable ctor reading a nonzero msg.value must deploy with `{value:}`.
  const bool ctor_needs_value =
    ctor_reads_value && ctor_is_payable && ctor_value_nonzero;
  // A NON-payable ctor cannot legally receive value in the EVM, yet ESBMC's model
  // seeds a nondet deploy msg.value regardless, so it may produce a construction
  // counterexample that relies on a nonzero value (e.g. a contradictory
  // `require(msg.value > 0)` in a non-payable ctor). Forge cannot reproduce it —
  // `new C()` sends 0 and setUp reverts — so mark that deploy UNSUPPORTED rather
  // than emit a guaranteed-reverting test.
  const bool ctor_value_unsendable =
    ctor_reads_value && !ctor_is_payable && ctor_value_nonzero;
  const bool ctor_needs_env = ctor_reads_timestamp || ctor_needs_deployer ||
                              ctor_needs_value || ctor_value_unsendable;
  bool ctor_env_attached = false;
  auto attach_ctor_env = [&](sol_call &cc) {
    // ③A0 ctor-time env: carry deploy-time ambient so setUp can vm.warp /
    // vm.startPrank / vm.deal+{value:} before `new C()` when the constructor
    // reads block.timestamp / msg.sender / msg.value.
    if (ctor_reads_timestamp)
    {
      cc.block_timestamp = ctor_block_timestamp;
      cc.warp = true;
    }
    if (ctor_needs_deployer)
    {
      cc.msg_sender = ctor_msg_sender;
      cc.deployer = true;
    }
    if (ctor_needs_value)
    {
      cc.msg_value = ctor_msg_value;
      cc.payable = true;
    }
  };
  for (const auto &kv : ctor_args)
  {
    sol_call cc = build_call(kv.first, kv.first, kv.second);
    if (ctor_needs_env && kv.first == ctor_ts_contract)
    {
      attach_ctor_env(cc);
      // Non-payable ctor needing a nonzero deploy value: degrade to UNSUPPORTED.
      if (ctor_value_unsendable)
      {
        cc.supported = false;
        cc.ctor_value_unsendable = true;
      }
      ctor_env_attached = true;
    }
    // A base-forwarded (remapped) deploy ctor gets its recovered args from a
    // base, but any param not matched there falls to a type DEFAULT. A defaulted
    // ctor arg can violate a ctor `require` and revert setUp (e.g. St1inch's
    // `feeReceiver_ = address(0)` / a zero `expBase_`), so degrade to UNSUPPORTED
    // unless EVERY arg is recovered or a mock. (EscrowDst's args are all
    // recovered/mock, so it still renders.)
    if (deploy_remapped && kv.first == ctor_ts_contract)
    {
      if (std::any_of(cc.args.begin(), cc.args.end(), [](const sol_arg &a) {
            return a.defaulted;
          }))
      {
        cc.supported = false;
        cc.ctor_unrecovered = true;
      }
      else
        cc.ctor_remapped = true;
    }
    calls.push_back(std::move(cc));
  }
  // No ctor_args entry for the deploy contract, yet its constructor reads
  // deploy-time env: synthesize a ctor call to carry the vm.warp/startPrank.
  // Route through build_call (not a bare no-arg `sol_call`) so a PARAMETERIZED
  // constructor never renders as an uncompilable `new C()`. A truly
  // parameterless ctor yields `new C()`; a parameterized ctor reached here has
  // NO recovered arguments, so rather than emit all-default args (which a ctor
  // `require` on a zero default could revert, breaking the whole suite) the
  // deploy degrades to UNSUPPORTED.
  if (ctor_needs_env && !ctor_env_attached && !ctor_ts_contract.empty())
  {
    sol_call cc = build_call(ctor_ts_contract, ctor_ts_contract, {});
    // Reached with NO recovered args, so build_call filled every slot with a
    // mock (interface) or a type DEFAULT. A default (0/address(0)) is a guess a
    // ctor `require` could revert (breaking setUp), so degrade to UNSUPPORTED —
    // UNLESS every argument is a faithful mock instance (or the ctor is
    // parameterless, giving the correct `new C()`).
    const bool all_mock_or_none =
      std::all_of(cc.args.begin(), cc.args.end(), [](const sol_arg &a) {
        return !a.mock_iface.empty();
      });
    if (!all_mock_or_none)
    {
      cc.supported = false;
      cc.ctor_unrecovered = true;
    }
    attach_ctor_env(cc);
    if (ctor_value_unsendable)
    {
      cc.supported = false;
      cc.ctor_value_unsendable = true;
    }
    calls.push_back(std::move(cc));
  }
  for (const auto &s : segs)
    if (!s.method.empty()) // dispatcher chose no method on this path
    {
      sol_call c = build_call(s.contract, s.method, s.args);
      c.reverts = s.reverts;
      c.msg_value = s.msg_value; // ③A0: env pin (emitted only when c.payable)
      c.block_timestamp = s.block_timestamp;
      c.warp = s.reads_timestamp; // ③A0: warp only when the body reads time
      c.msg_sender = s.msg_sender;
      // ③A0: pin the per-tx sender when it is reproducible (top-level, not
      // overwritten by a nested call) AND relevant — either the covered body
      // reads msg.sender directly, or the contract's ctor bound an owner
      // (`owner = msg.sender`), in which case ANY method may sit behind an
      // `onlyOwner` modifier whose guard is on the PATH to the covered branch
      // (not in the branch's own condition), so the sender must match the
      // deployer for the call to reach that branch rather than revert.
      c.prank = !s.sender_dirty && s.msg_sender &&
                (s.reads_msg_sender || ctor_needs_deployer);
      calls.push_back(std::move(c));
    }

  // Coverage-claim fallback: the normal paths produced no call (no --function,
  // no reconstructed ctor, and no dispatcher segment got a method). Under
  // per-claim slicing the dispatcher's FIRST tx-guard ($1) can be sliced away
  // when the covered method is selected by a LATER guard, or a loop-only branch
  // never sets the segment method (e.g. dock/push) — so the segment path yields
  // nothing. Derive the covered method directly from the guard-true coverage
  // assert's source location and reconstruct one call to it, with recovered
  // params (or type defaults for dispatcher-supplied nondet args).
  if (calls.empty() && focus_fn.empty())
  {
    // Contracts that expose a dispatcher (`_ESBMC_Nondet_Extcall_<C>`): the
    // covered method's owner is whichever of these can call it.
    std::set<std::string> disp_contracts;
    ns.get_context().foreach_operand([&](const symbolt &s) {
      const std::string id = s.id.as_string();
      const std::string mark = "@_ESBMC_Nondet_Extcall_";
      size_t p = id.find(mark);
      if (
        s.type.is_code() && has_prefix(id, "sol:@C@") && p != std::string::npos)
        disp_contracts.insert(id.substr(7, p - 7));
    });

    for (auto const &step : target.SSA_steps)
    {
      if (!step.is_assert() || !smt_conv.l_get(step.guard_ast).is_true())
        continue;
      // The coverage assert's source location names the covered method (bare
      // name, e.g. "dock", or a modifier wrapper "bump_onlyOwner"); resolve it
      // to the real dispatcher-callable method and its owner contract.
      const std::string raw_m = step_location_method(step);
      if (raw_m.empty())
        continue;
      std::string c = config.options.get_option("contract");
      std::string m =
        c.empty() ? std::string() : resolve_dispatcher_method(c, raw_m);
      if (m.empty())
      {
        c.clear();
        for (const auto &cand : disp_contracts)
        {
          const std::string r = resolve_dispatcher_method(cand, raw_m);
          if (!r.empty())
          {
            c = cand;
            m = r;
            break;
          }
        }
      }
      if (c.empty())
        continue;
      std::map<std::string, sol_arg> recovered;
      for (const auto &decl : get_method_params(ns, c, m))
      {
        expr2tc v = recover_focus_param(target, smt_conv, c, m, decl.first);
        if (!v)
          continue;
        sol_arg a;
        a.param = decl.first;
        a.sol_type = decl.second;
        a.value = v;
        a.literal = format_sol_value(decl.second, v);
        recovered[decl.first] = a;
      }
      sol_call call = build_call(c, m, recovered);
      if (call.supported)
      {
        calls.push_back(std::move(call));
        break;
      }
    }
  }

  std::set<std::string> used;
  for (const auto &c : calls)
    used.insert(c.contract);

  // Drop base contracts: only the most-derived contract is instantiated. A
  // base contract's constructor runs as part of `new Derived(...)`, so a
  // recovered base ctor (or a call whose body ran inside the derived instance)
  // must NOT become its own `new Base(...)` — Base is often abstract and, even
  // when concrete, is a duplicate instance. Remove every call whose contract is
  // a base of another used contract.
  std::set<std::string> is_base;
  for (const auto &cn : used)
    for (const auto &b : contract_bases(ns, cn))
      if (b != cn && used.count(b))
        is_base.insert(b);
  if (!is_base.empty())
  {
    calls.erase(
      std::remove_if(
        calls.begin(),
        calls.end(),
        [&](const sol_call &c) { return is_base.count(c.contract) != 0; }),
      calls.end());
    used.clear();
    for (const auto &c : calls)
      used.insert(c.contract);
  }

  // Synthesise a defaulted constructor for any contract that was called but
  // whose parameterized constructor was not reconstructed, so `new C(...)`
  // still compiles (parameterless constructors need no call: `new C()`).
  std::set<std::string> has_ctor;
  for (const auto &c : calls)
    if (c.method == c.contract)
      has_ctor.insert(c.contract);
  // Record which touched contracts cannot be `new`'d (abstract/interface/
  // library); write_foundry_file degrades their instantiation to UNSUPPORTED.
  for (const auto &cn : used)
    if (contract_is_non_instantiable(ns, cn))
      non_instantiable.insert(cn);
  // A library is called statically (`Lib.fn(args)`, no instance) rather than
  // constructed — record it so write_foundry_file emits the static form.
  for (const auto &cn : used)
    if (contract_is_library(ns, cn))
      libraries.insert(cn);
  for (const auto &cn : used)
    if (!has_ctor.count(cn) && !get_method_params(ns, cn, cn).empty())
      calls.push_back(build_call(cn, cn, {}));

  // Constructors precede the transactions that use their instance.
  std::stable_partition(calls.begin(), calls.end(), [](const sol_call &c) {
    return c.method == c.contract;
  });
  return calls;
}

// ---------------------------------------------------------------------------
// Collection
// ---------------------------------------------------------------------------

void foundry_generator::clear()
{
  std::lock_guard<std::mutex> lock(data_mutex);
  test_cases.clear();
  source_file.clear();
  non_instantiable.clear();
  libraries.clear();
  mock_specs.clear();
}

void foundry_generator::collect(
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  const namespacet &ns)
{
  test_case tc = reconstruct(target, smt_conv, ns);
  if (tc.empty())
    return;

  std::lock_guard<std::mutex> lock(data_mutex);
  if (source_file.empty())
    source_file = config.options.get_option("input-file");
  test_cases.push_back(std::move(tc));
}

bool foundry_generator::has_tests() const
{
  std::lock_guard<std::mutex> lock(data_mutex);
  return !test_cases.empty();
}

std::string foundry_generator::fingerprint(const test_case &tc)
{
  std::string fp;
  for (const auto &call : tc)
  {
    if (call.reverts)
      fp += "revert:";
    fp += call.contract;
    fp += '.';
    fp += call.method;
    fp += '(';
    for (const auto &a : call.args)
    {
      fp += a.literal.empty() ? "?" : a.literal;
      fp += ',';
    }
    fp += ")";
    // ③A0: two counterexamples that differ ONLY in msg.value (not a param) must
    // NOT dedup to one case — fold the emitted value into the fingerprint.
    if (call.payable && call.msg_value)
    {
      const std::string v = format_sol_value("UINT256", call.msg_value);
      if (!v.empty() && v != "0")
        fp += "{value:" + v + "}";
    }
    // ③A0: two CEs differing only in block.timestamp must not dedup either.
    if (call.warp && call.block_timestamp)
    {
      const std::string t = format_sol_value("UINT256", call.block_timestamp);
      if (!t.empty())
        fp += "[warp:" + t + "]";
    }
    // ③A0: the onlyOwner PASS arm (sender==owner) and FAIL arm (sender!=owner)
    // may share method+args — fold the pinned sender so they do not dedup.
    if (call.prank && call.msg_sender)
    {
      const std::string sdr = format_sol_value("ADDRESS", call.msg_sender);
      if (!sdr.empty())
        fp += "<prank:" + sdr + ">";
    }
    // ③A0: a ctor carrier's deployer (setUp vm.startPrank) distinguishes cases
    // that differ ONLY by owner — fold it so they don't dedup before grouping.
    if (call.deployer && call.msg_sender)
    {
      const std::string d = format_sol_value("ADDRESS", call.msg_sender);
      if (!d.empty())
        fp += "<dep:" + d + ">";
    }
    fp += ";";
  }
  return fp;
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

static std::string file_stem(const std::string &path)
{
  size_t slash = path.find_last_of("/\\");
  std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
  size_t dot = base.find('.');
  return dot == std::string::npos ? base : base.substr(0, dot);
}

size_t foundry_generator::write_foundry_file(
  const std::string &path,
  const std::string &primary,
  const std::vector<test_case> &cases) const
{
  // Count of calls wrapped in revert-tolerant try/catch (returned so the caller
  // can report the silent-tolerance count, mirroring the vm.expectRevert line).
  size_t revert_tolerant = 0;

  auto join_args = [](const sol_call &call) {
    std::string s;
    for (size_t i = 0; i < call.args.size(); ++i)
      s += (i ? ", " : "") + call.args[i].literal;
    return s;
  };

  // A construction plan: one instance per distinct contract (sorted for a
  // stable var mapping shared across the group), built via its reconstructed
  // constructor call (or `new C()` when parameterless). A single dispatcher
  // drives one `_ESBMC_Object_<C>` across all txs, so one instance suffices.
  struct inst
  {
    std::string contract, var, ctor_args;
    bool buildable;
    bool abstract;         // non-instantiable (abstract/interface/library)
    std::string ctor_warp; // ③A0: vm.warp timestamp for a time-dependent ctor
    std::string deployer; // ③A0: vm.startPrank deployer for a sender-owner ctor
    std::string ctor_value; // ③A0: {value:} for a payable value-reading ctor
    bool value_unsendable =
      false; // non-payable ctor needs value -> unsupported
    bool unrecovered =
      false; // parameterized ctor, args not recovered -> unsupported
  };
  auto plan_of = [&](const test_case &tc) {
    std::map<std::string, const sol_call *> ctor;
    std::set<std::string> used;
    for (const auto &c : tc)
    {
      // Libraries are never instantiated (called statically); keep them out of
      // the construction plan entirely.
      if (libraries.count(c.contract))
        continue;
      used.insert(c.contract);
      if (c.method == c.contract)
        ctor[c.contract] = &c;
    }
    std::vector<inst> plan;
    for (const auto &cn : used)
    {
      inst ib;
      ib.contract = cn;
      ib.var = "c" + std::to_string(plan.size());
      ib.abstract = non_instantiable.count(cn) != 0;
      auto it = ctor.find(cn);
      // Solidity forbids `new` on abstract/interface/library contracts, so they
      // are never buildable regardless of whether their ctor args rendered.
      ib.buildable =
        !ib.abstract && (it == ctor.end() || it->second->supported);
      if (it != ctor.end() && it->second->ctor_value_unsendable)
        ib.value_unsendable = true;
      if (it != ctor.end() && it->second->ctor_unrecovered)
        ib.unrecovered = true;
      ib.ctor_args = it == ctor.end() ? std::string() : join_args(*it->second);
      // ③A0 ctor-time env: a ctor reading block.timestamp must be deployed under
      // a matching vm.warp, else `new C()` reverts in setUp and fails the suite.
      if (it != ctor.end() && it->second->warp && it->second->block_timestamp)
      {
        const std::string t =
          format_sol_value("UINT256", it->second->block_timestamp);
        if (!t.empty())
          ib.ctor_warp = t;
      }
      // ③A0 ctor-time deployer: a ctor binding `owner = msg.sender` must deploy
      // under a known identity so an `onlyOwner` call can match (or mismatch) it.
      if (it != ctor.end() && it->second->deployer && it->second->msg_sender)
      {
        const std::string d =
          format_sol_value("ADDRESS", it->second->msg_sender);
        if (!d.empty())
          ib.deployer = d;
      }
      // ③A0 ctor-time value: a payable ctor reading msg.value deploys with
      // `new C{value: v}(...)` (funded via vm.deal) so a value require passes.
      if (it != ctor.end() && it->second->payable && it->second->msg_value)
      {
        const std::string v =
          format_sol_value("UINT256", it->second->msg_value);
        if (!v.empty() && v != "0")
          ib.ctor_value = v;
      }
      plan.push_back(ib);
    }
    return plan;
  };
  auto sig_of = [&](const std::vector<inst> &plan) {
    std::string s;
    for (const auto &ib : plan)
    {
      s += ib.contract + "(" + (ib.buildable ? ib.ctor_args : "!") + ")";
      // ③A0: a ctor whose initialized state depends on the deploy timestamp must
      // not share one setUp warp across cases needing different timestamps.
      if (!ib.ctor_warp.empty())
        s += "@warp" + ib.ctor_warp;
      // ③A0: a different deploy sender means a different owner — cases needing
      // distinct owners must not share one setUp startPrank.
      if (!ib.deployer.empty())
        s += "@dep" + ib.deployer;
      // ③A0: a ctor whose state depends on deploy msg.value must not share a
      // setUp funded with a different value.
      if (!ib.ctor_value.empty())
        s += "@val" + ib.ctor_value;
      s += ";";
    }
    return s;
  };

  // Group cases by construction signature, preserving first-seen order.
  std::vector<std::string> group_order;
  std::map<std::string, std::vector<const test_case *>> groups;
  for (const auto &tc : cases)
  {
    std::string s = sig_of(plan_of(tc));
    if (!groups.count(s))
      group_order.push_back(s);
    groups[s].push_back(&tc);
  }

  std::set<std::string> imports;
  std::set<std::string> mock_ifaces_used; // interfaces needing an ESBMCMock_*
  for (const auto &tc : cases)
    for (const auto &call : tc)
    {
      imports.insert(call.contract);
      // A user-defined value type used as an argument ("UDVT:<Name>:...") must
      // be imported for `Name.wrap(...)` to resolve — but only when it is a
      // file-level type (a contract-scoped UDVT is named `Scope.Name` and its
      // enclosing contract is already imported).
      for (const auto &a : call.args)
      {
        if (has_prefix(a.sol_type, "UDVT:"))
        {
          std::string name = a.sol_type.substr(5);
          name = name.substr(0, name.find(':'));
          if (name.find('.') == std::string::npos)
            imports.insert(name);
        }
        // A struct literal `<Scope>.<Name>(…)` needs `<Scope>` imported; a
        // top-level struct `<Name>(…)` needs `<Name>` itself. Plus any type the
        // literal's fields reference (UDVT names, nested struct scopes).
        if (has_prefix(a.sol_type, "STRUCT:"))
        {
          const std::string name = a.sol_type.substr(7);
          size_t dot = name.find('.');
          imports.insert(dot == std::string::npos ? name : name.substr(0, dot));
        }
        for (const auto &si : a.struct_imports)
          imports.insert(si);
        // An interface-arg mock `is <iface>` must import <iface> from the source.
        if (!a.mock_iface.empty())
        {
          imports.insert(a.mock_iface);
          mock_ifaces_used.insert(a.mock_iface);
        }
      }
    }

  // Import from the Solidity source (a `.sol` file), never the `.solast` AST
  // input: forge compiles `.sol`, and the two share a directory. In the
  // two-arg invocation `--sol contract.solast contract.sol`, `--sol` holds the
  // `.solast`, so prefer whichever candidate actually ends in `.sol`.
  auto ends_with = [](const std::string &s, const std::string &suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
  };
  std::string sol_opt = config.options.get_option("sol");
  std::string src_base =
    (!source_file.empty() && ends_with(source_file, ".sol")) ? source_file
    : ends_with(sol_opt, ".sol")                             ? sol_opt
    : !source_file.empty()                                   ? source_file
                                                             : sol_opt;
  size_t slash = src_base.find_last_of("/\\");
  if (slash != std::string::npos)
    src_base = src_base.substr(slash + 1);

  std::ofstream f(path);
  f << "// SPDX-License-Identifier: MIT\n";
  f << "// Auto-generated by ESBMC " << ESBMC_VERSION << "\n";
  f << "// Foundry coverage test reconstructed from ESBMC counterexamples.\n";
  f << "// Calls marked [revert-tolerant] are wrapped in try/catch: ESBMC "
       "could\n";
  f << "// not confirm the call's outcome, so an undetected revert (e.g.\n";
  f << "// require(cond, CustomError)) does not FAIL the assertion-free "
       "replay.\n";
  f << "// The wrap does NOT assert a revert; a non-reverting call runs "
       "normally.\n";
  f << "pragma solidity >=0.8.0;\n\n";
  f << "import {Test} from \"forge-std/Test.sol\";\n";
  for (const auto &c : imports)
    f << "import {" << c << "} from \"./" << src_base << "\";\n";

  // Interface-arg mocks: emit one `contract ESBMCMock_<iface> is <iface>` with a
  // default-returning stub per interface method, so a constructor/method taking
  // an `<iface>` handle can be deployed. Its calls to the handle return fixed
  // defaults — reproducing ESBMC's havoc of those calls for every branch except
  // one that depends on a specific return value ([approx], noted below).
  for (const auto &iface : mock_ifaces_used)
  {
    auto it = mock_specs.find(iface);
    if (it == mock_specs.end() || !it->second.renderable)
      continue; // should not happen (only renderable ifaces reach here)
    f << "\n// [approx] mock for interface " << iface
      << ": all methods return fixed defaults; branches on its return values "
         "are not reproduced.\n";
    f << "contract ESBMCMock_" << iface << " is " << iface << " {\n";
    for (const auto &stub : it->second.stubs)
      f << stub << "\n";
    f << "}\n";
  }

  const bool multi = group_order.size() > 1;
  size_t gidx = 0, fn = 0;
  for (const auto &s : group_order)
  {
    const auto &grp = groups[s];
    const auto plan = plan_of(*grp.front());
    std::map<std::string, std::string> var;
    std::set<std::string> built;
    for (const auto &ib : plan)
    {
      var[ib.contract] = ib.var;
      if (ib.buildable)
        built.insert(ib.contract);
    }

    // Interface-arg mock instances this group deploys: (var name, iface),
    // unique in first-seen order. The var names already appear in the ctor/method
    // argument literals (`mk_<iface>_<key>`), so a distinct alias key => a
    // distinct instance (reproducing `a != b`), a shared key => one instance.
    std::vector<std::pair<std::string, std::string>> mock_insts;
    {
      std::set<std::string> seen;
      for (const auto *tcp : grp)
        for (const auto &call : *tcp)
        {
          // An UNSUPPORTED call is not emitted, so its mock arguments would be
          // orphaned (deployed but never passed). Skip them.
          if (!call.supported)
            continue;
          for (const auto &a : call.args)
            if (
              !a.mock_iface.empty() && !a.literal.empty() &&
              seen.insert(a.literal).second)
              mock_insts.emplace_back(a.literal, a.mock_iface);
        }
    }

    f << "\ncontract " << primary << "CovTest";
    if (multi)
      f << "_" << gidx;
    f << " is Test {\n";

    // State-variable instances, deployed once in setUp() — Foundry re-runs
    // setUp() before every test_cov_*, giving each a fresh construction.
    for (const auto &mi : mock_insts)
      f << "  ESBMCMock_" << mi.second << " " << mi.first << ";\n";
    for (const auto &ib : plan)
      if (ib.buildable)
        f << "  " << ib.contract << " " << ib.var << ";\n";
    f << "  function setUp() public {\n";
    // Deploy interface-arg mocks first: the contract constructors below receive
    // them (a bare address would revert when the constructor calls a method).
    for (const auto &mi : mock_insts)
      f << "    " << mi.first << " = new ESBMCMock_" << mi.second << "();\n";
    // ③A0 ctor-time env: if any deployed contract's constructor reads
    // block.timestamp, vm.warp to the deploy-time value BEFORE constructing, so
    // a `require(block.timestamp ...)` in the ctor does not revert setUp and fail
    // the whole suite. One warp suffices (the per-tx warps in test_cov_* are
    // monotonically >= this, matching ESBMC's non-decreasing block.timestamp).
    for (const auto &ib : plan)
      if (ib.buildable && !ib.ctor_warp.empty())
      {
        f << "    vm.warp(" << ib.ctor_warp << ");\n";
        break;
      }
    for (const auto &ib : plan)
      if (ib.buildable)
      {
        // ③A0 ctor-time deployer: deploy under a known msg.sender so a
        // `owner = msg.sender` binding pins an identity a later onlyOwner call
        // can match (or mismatch) to cover both modifier arms.
        if (!ib.deployer.empty())
          f << "    vm.startPrank(" << ib.deployer << ");\n";
        // ③A0 ctor-time value: fund the test and forward `{value: v}` so a
        // payable ctor's `require(msg.value ...)` passes instead of reverting
        // setUp. Only emitted for a payable ctor (see reconstruct()).
        std::string value_brace;
        if (!ib.ctor_value.empty())
        {
          f << "    vm.deal(address(this), " << ib.ctor_value << ");\n";
          value_brace = "{value: " + ib.ctor_value + "}";
        }
        f << "    " << ib.var << " = new " << ib.contract << value_brace << "("
          << ib.ctor_args << ");\n";
        if (!ib.deployer.empty())
          f << "    vm.stopPrank();\n";
      }
      else if (ib.abstract)
        f << "    // UNSUPPORTED: " << ib.contract
          << " is abstract / an interface / a library and cannot be "
             "instantiated with `new`\n";
      else if (ib.value_unsendable)
        f << "    // UNSUPPORTED: constructor of " << ib.contract
          << " requires a nonzero deploy-time msg.value but is not payable "
             "(EVM forbids sending value); the reconstructed deploy path is "
             "not reproducible in Foundry\n";
      else if (ib.unrecovered)
        f << "    // UNSUPPORTED: constructor of " << ib.contract
          << " has parameters but its arguments were not recovered on this "
             "path "
             "(e.g. --focus-function nondets them); deploying with default "
             "arguments could revert setUp, so the deploy is skipped\n";
      else
        f << "    // UNSUPPORTED: constructor of " << ib.contract
          << " has an argument type ESBMC cannot yet render as a literal\n";
    f << "  }\n";

    for (const auto *tcp : grp)
    {
      f << "  function test_cov_" << fn++ << "() public {\n";
      for (const auto &call : *tcp)
      {
        if (call.method == call.contract)
          continue; // constructor -> setUp()
        // A library call has no instance: emit the static receiver `Lib`. A
        // contract call uses its constructed instance variable.
        const bool is_lib = libraries.count(call.contract) != 0;
        const std::string recv = is_lib ? call.contract : var[call.contract];
        // ③A0 environment pinning: for a payable method whose transaction the
        // solver gave a non-zero msg.value, reproduce it — `vm.deal` funds the
        // test contract, `{value: N}` forwards it. Non-payable methods never
        // get a value (sending value to them reverts). This is faithful (it
        // replays exactly the ESBMC-chosen value) and self-contained (no deploy
        // coordination).
        std::string value_brace, deal_line;
        if (call.payable && call.msg_value)
        {
          const std::string v = format_sol_value("UINT256", call.msg_value);
          if (!v.empty() && v != "0")
          {
            deal_line = "    vm.deal(address(this), " + v + ");\n";
            value_brace = "{value: " + v + "}";
          }
        }
        // ③A0: block.timestamp pin — vm.warp before the call when the covered
        // path reads block.timestamp. Prepended to deal_line so both env pins
        // are emitted together, only in the branches that emit a call.
        if (call.warp && call.block_timestamp)
        {
          const std::string t =
            format_sol_value("UINT256", call.block_timestamp);
          if (!t.empty())
            deal_line = "    vm.warp(" + t + ");\n" + deal_line;
        }
        // ③A0: msg.sender pin — vm.prank sets the sender for the NEXT call ONLY,
        // so it must be the last cheatcode before the call. Appended after
        // warp/deal, emitted only in call-emitting branches (and only for a
        // top-level sender-clean read — see reconstruct()).
        if (call.prank && call.msg_sender)
        {
          const std::string sdr = format_sol_value("ADDRESS", call.msg_sender);
          if (!sdr.empty())
            deal_line += "    vm.prank(" + sdr + ");\n";
        }

        if (!call.supported || (!is_lib && !built.count(call.contract)))
          // No `vm.deal` here: the call is not emitted, so an orphan deal would
          // be dead noise and would over-report the pinned-value count.
          f << "    // UNSUPPORTED: " << call.contract << "." << call.method
            << " has an argument type ESBMC cannot yet render as a literal\n";
        else if (call.reverts)
        {
          // Phase A detected this edge reverts (conservative #sol_error
          // straight-line). Assert the revert precisely with bare
          // vm.expectRevert() so an ESBMC<->forge divergence (it does NOT
          // revert) surfaces as a loud test failure rather than being hidden.
          f << deal_line;
          f << "    vm.expectRevert();\n";
          f << "    " << recv << "." << call.method << value_brace << "("
            << join_args(call) << ");\n";
        }
        else if (is_lib)
        {
          // A library call inlines into the test, so it cannot be wrapped in
          // try/catch (Solidity allows try/catch only on external calls). It is
          // called directly. This is sound for the reconstructed input: a
          // library `require`/`revert` lowers to `__ESBMC_assume`, so the
          // reverting branch is pruned and can never be the covered branch —
          // the recovered argument always drives a non-reverting path. (A
          // revert from arithmetic/an external call the library itself makes is
          // the same residual any call kind carries.)
          f << deal_line;
          f << "    " << recv << "." << call.method << value_brace << "("
            << join_args(call) << ");\n";
        }
        else
        {
          // Outcome not confirmed: wrap in try/catch so an undetected revert
          // (e.g. require(cond, CustomError), which lowers with no detectable
          // terminator) does not FAIL the assertion-free replay. A
          // non-reverting call runs normally in the try body. This is the only
          // place a revert is silently tolerated (see file header).
          f << deal_line;
          f << "    // [revert-tolerant] outcome not asserted\n";
          f << "    try " << recv << "." << call.method << value_brace << "("
            << join_args(call) << ") {} catch {}\n";
          ++revert_tolerant;
        }
      }
      f << "  }\n";
    }
    f << "}\n";
    ++gidx;
  }
  return revert_tolerant;
}

// One file per contract-under-test; write_foundry_file then splits each file's
// cases into per-construction test contracts.
void foundry_generator::generate() const
{
  std::lock_guard<std::mutex> lock(data_mutex);
  if (test_cases.empty())
  {
    log_warning("No Foundry test cases collected. No *.t.sol generated.");
    return;
  }

  // Deduplicate structurally-identical reconstructions.
  std::unordered_set<std::string> seen;
  std::vector<test_case> unique;
  for (const auto &tc : test_cases)
    if (seen.insert(fingerprint(tc)).second)
      unique.push_back(tc);

  // Group by contract-under-test, preserving first-seen order.
  std::vector<std::string> order;
  std::map<std::string, std::vector<test_case>> by_primary;
  for (const auto &tc : unique)
  {
    std::string p = primary_contract(tc);
    if (p.empty())
      p = file_stem(source_file);
    if (!by_primary.count(p))
      order.push_back(p);
    by_primary[p].push_back(tc);
  }

  for (const auto &p : order)
  {
    const auto &cs = by_primary[p];
    std::string path = p + ".cov.t.sol";
    size_t revert_tolerant = write_foundry_file(path, p, cs);
    log_status(
      "Generated Foundry coverage test with {} case(s): {}", cs.size(), path);
    // Interface-arg mock synthesis: report the distinct interfaces for which an
    // `ESBMCMock_*` was emitted (their calls return fixed defaults — [approx]).
    std::set<std::string> mifaces;
    for (const auto &tc : cs)
      for (const auto &call : tc)
        for (const auto &a : call.args)
          if (!a.mock_iface.empty())
            mifaces.insert(a.mock_iface);
    if (!mifaces.empty())
      log_status(
        "Foundry: {} interface mock(s) synthesized (ESBMCMock_*, [approx] "
        "fixed-default returns)",
        mifaces.size());
    // Struct-typed arguments rendered as positional struct literals.
    size_t struct_args = 0;
    for (const auto &tc : cs)
      for (const auto &call : tc)
        for (const auto &a : call.args)
          if (has_prefix(a.sol_type, "STRUCT:"))
            ++struct_args;
    if (struct_args)
      log_status("Foundry: {} struct-literal arg(s) rendered", struct_args);
    // Deploy ctors whose args were recovered under a base ctor and remapped.
    size_t remapped = 0;
    for (const auto &tc : cs)
      for (const auto &call : tc)
        if (call.ctor_remapped)
          ++remapped;
    if (remapped)
      log_status(
        "Foundry: {} deploy(s) with base-forwarded constructor args", remapped);
    // A contract/interface-typed argument that was NOT mocked (a concrete
    // contract, an interface whose full stub set could not render, or one not
    // materialized in the symbol table) degraded to UNSUPPORTED — report it so
    // the coverage gap is visible rather than silent.
    std::set<std::string> degraded;
    for (const auto &tc : cs)
      for (const auto &call : tc)
        for (const auto &a : call.args)
          if (has_prefix(a.sol_type, "CONTRACT:") && a.mock_iface.empty())
            degraded.insert(a.sol_type.substr(9));
    if (!degraded.empty())
      log_status(
        "Foundry: {} contract-typed arg type(s) UNSUPPORTED (not mocked: "
        "concrete/unrenderable/absent handle)",
        degraded.size());
    // A parameterized constructor whose args were not recovered on this path
    // (e.g. --focus-function) → deploy degraded to UNSUPPORTED rather than emit
    // default args that could revert setUp. Report it so the gap is visible.
    std::set<std::string> unrec;
    for (const auto &tc : cs)
      for (const auto &call : tc)
        if (call.ctor_unrecovered)
          unrec.insert(call.contract);
    if (!unrec.empty())
      log_status(
        "Foundry: {} deploy(s) UNSUPPORTED (constructor args not recovered on "
        "this path)",
        unrec.size());
    // A DETECTED reverting edge (Phase A: conservative #sol_error straight-line)
    // is wrapped in a precise vm.expectRevert(); report both counts so the
    // wrapping (and the silently revert-tolerant try/catch fallback) is visible.
    size_t revert_cases = 0;
    for (const auto &tc : cs)
      for (const auto &call : tc)
        if (call.reverts)
          ++revert_cases;
    if (revert_cases)
      log_status(
        "Foundry: {} call(s) wrapped in vm.expectRevert (detected revert edge)",
        revert_cases);
    if (revert_tolerant)
      log_status(
        "Foundry: {} call(s) wrapped in revert-tolerant try/catch "
        "(outcome not confirmed)",
        revert_tolerant);
    // ③A0: report payable calls whose msg.value was pinned (vm.deal + {value:}),
    // so the environment reconstruction is visible/assertable.
    size_t value_pinned = 0, time_pinned = 0, sender_pinned = 0;
    for (const auto &tc : cs)
      for (const auto &call : tc)
      {
        if (!call.supported)
          continue;
        const bool is_ctor = call.method == call.contract;
        if (call.payable && call.msg_value)
        {
          const std::string v = format_sol_value("UINT256", call.msg_value);
          if (!v.empty() && v != "0")
            ++value_pinned;
        }
        // A ctor carrier (method==contract) warps in setUp, not as a call.
        if (call.warp && call.block_timestamp && !is_ctor)
        {
          const std::string t =
            format_sol_value("UINT256", call.block_timestamp);
          if (!t.empty())
            ++time_pinned;
        }
        // Per-call sender pins (vm.prank); a ctor carrier's deployer pin is a
        // setUp vm.startPrank, not a call, so it is excluded here.
        if (call.prank && call.msg_sender && !is_ctor)
        {
          const std::string s = format_sol_value("ADDRESS", call.msg_sender);
          if (!s.empty())
            ++sender_pinned;
        }
      }
    if (value_pinned)
      log_status(
        "Foundry: {} call(s) with pinned msg.value (payable env)",
        value_pinned);
    if (time_pinned)
      log_status(
        "Foundry: {} call(s) with pinned block.timestamp (vm.warp)",
        time_pinned);
    if (sender_pinned)
      log_status(
        "Foundry: {} call(s) with pinned msg.sender (vm.prank)", sender_pinned);
  }
}

void foundry_generator::generate_single(
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  const namespacet &ns)
{
  if (source_file.empty())
    source_file = config.options.get_option("input-file");

  test_case tc = reconstruct(target, smt_conv, ns);
  if (tc.empty())
  {
    log_warning(
      "No reconstructable transaction found. No Foundry test generated.");
    return;
  }

  std::string p = primary_contract(tc);
  if (p.empty())
    p = file_stem(source_file);
  std::string path = p + ".cov.t.sol";
  write_foundry_file(path, p, {tc});
  log_status("Generated Foundry test: {}", path);
}

std::string foundry_generator::primary_contract(const test_case &tc)
{
  for (const auto &c : tc)
    if (c.method != c.contract)
      return c.contract;
  return tc.empty() ? std::string() : tc.front().contract;
}
