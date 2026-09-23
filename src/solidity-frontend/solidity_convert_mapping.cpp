/// \file solidity_convert_mapping.cpp
/// \brief Mapping type conversion for the Solidity frontend.
///
/// Converts Solidity mapping types into ESBMC's representation using
/// infinite arrays (modeled as arrays with nondet size). Handles nested
/// mappings, mapping access expressions, and the generation of helper
/// symbols for mapping state variables.

#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_expr.h>
#include <util/message.h>
#include <fstream>

void solidity_convertert::get_mapping_inf_arr_name(
  const std::string &cname,
  const std::string &name,
  std::string &arr_name,
  std::string &arr_id)
{
  arr_name = "_ESBMC_inf_" + name;
  // we cannot define a mapping inside a function body
  arr_id = "sol:@C@" + cname + "@" + arr_name + "#";
}

// Builds the canonical `{ base, mid, addr }` mapping_t initializer used by
// every per-instance mapping init.  The path_name is what disambiguates
// the global pool — top-level state vars pass the field name (`m`),
// struct-internal mappings pass a "_"-joined path (`bx_m`,
// `outer_inner_m`), so each mapping field gets its own
// `_ESBMC_inf_<cname>_<path_name>[]` backing array.  The global is created
// lazily and cached via `context.find_symbol` so nested-mapping inits
// emitted from the Phase-2 ctor walker don't duplicate state-var-decl
// allocations.  next_mapping_mid is bumped on every call — even if the
// same mapping field would be re-initialized in two different ctors,
// each gets a distinct `mid`, but they share the global pool by name.
bool solidity_convertert::build_mapping_t_init_value(
  const std::string &cname,
  const std::string &path_name,
  const exprt &addr_owner_this,
  const locationt &loc,
  exprt &out_init)
{
  std::string arr_name, arr_id;
  get_mapping_inf_arr_name(cname, path_name, arr_name, arr_id);

  std::string mapping_struct_name = "_ESBMC_Mapping";
  if (context.find_symbol(lib_prefix + mapping_struct_name) == nullptr)
  {
    log_error("failed to find _ESBMC_Mapping reference");
    return true;
  }

  typet arr_t = array_typet(
    symbol_typet(lib_prefix + mapping_struct_name), exprt("infinity"));

  if (context.find_symbol(arr_id) == nullptr)
  {
    symbolt arr_s;
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    get_default_symbol(arr_s, debug_modulename, arr_t, arr_name, arr_id, loc);
    arr_s.static_lifetime = true;
    arr_s.file_local = true;
    arr_s.lvalue = true;
    auto &add_added_s = *move_symbol_to_context(arr_s);
    add_added_s.value = gen_zero(get_complete_type(arr_t, ns), true);
  }

  const symbolt *arr_sym = context.find_symbol(arr_id);
  if (arr_sym == nullptr)
  {
    log_error("failed to find/create mapping inf-array global {}", arr_id);
    return true;
  }

  typet map_t = context.find_symbol(lib_prefix + "mapping_t")->type;
  assert(map_t.is_struct());

  exprt inits = gen_zero(map_t);
  exprt op0 = symbol_expr(*arr_sym);

  const struct_typet &map_struct = to_struct_type(map_t);
  const auto &comps = map_struct.components();
  unsigned base_idx = 0, mid_idx = (unsigned)-1, addr_idx = 1;
  for (unsigned i = 0; i < comps.size(); i++)
  {
    if (comps[i].get_name() == "base")
      base_idx = i;
    else if (comps[i].get_name() == "mid")
      mid_idx = i;
    else if (comps[i].get_name() == "addr")
      addr_idx = i;
  }

  solidity_gen_typecast(ns, op0, comps[base_idx].type());
  inits.operands()[base_idx] = op0;

  if (mid_idx != (unsigned)-1)
  {
    exprt mid_expr =
      from_integer(next_mapping_mid++, comps[mid_idx].type());
    inits.operands()[mid_idx] = mid_expr;
  }

  exprt addr_expr = member_exprt(addr_owner_this, "$address", addr_t);
  solidity_gen_typecast(ns, addr_expr, comps[addr_idx].type());
  inits.operands()[addr_idx] = addr_expr;

  out_init = inits;
  return false;
}

/**
	@target: target index access child json
	return true if it's a mapping_set, including assign, assign+, tuple assign...
	otherwise return false, representing mapping_get
*/
bool solidity_convertert::is_mapping_set_lvalue(const nlohmann::json &target)
{
  assert(target.value("nodeType", "") == "IndexAccess");
  assert(target.contains("lValueRequested"));
  return target["lValueRequested"].get<bool>();
}

bool solidity_convertert::get_mapping_key_value_type(
  const nlohmann::json &map_node,
  typet &key_t,
  typet &value_t,
  SolidityGrammar::SolType &key_sol_type,
  SolidityGrammar::SolType &val_sol_type)
{
  assert(map_node.contains("typeName"));
  if (get_type_description(
        map_node["typeName"]["keyType"]["typeDescriptions"], key_t))
  {
    log_error("cannot get mapping key type");
    return true;
  }
  if (get_type_description(
        map_node["typeName"]["valueType"]["typeDescriptions"], value_t))
  {
    log_error("cannot get mapping value type");
    return true;
  }

  // set type flag
  key_sol_type = get_sol_type(key_t);
  val_sol_type = get_sol_type(value_t);
  if (val_sol_type == SolidityGrammar::SolType::UNSET)
    return true;
  return false;
}

// e.g. bytes3 = bytes3(x) ==> length == 3
void solidity_convertert::get_bytesN_size(
  const exprt &src_expr,
  exprt &len_expr)
{
  std::string byte_size = src_expr.type().get("#sol_bytesn_size").as_string();
  if (!byte_size.empty())
    len_expr = from_integer(std::stoul(byte_size), size_type());
  else
  {
    assert(!src_expr.is_nil());
    len_expr = member_exprt(src_expr, "length", size_type());
  }
}

bool solidity_convertert::get_dynamic_pool(
  const std::string &c_name,
  exprt &pool)
{
  exprt cur_this_expr;
  if (current_functionDecl)
  {
    if (get_func_decl_this_ref(*current_functionDecl, cur_this_expr))
      return true;
  }
  else
  {
    if (get_ctor_decl_this_ref(c_name, cur_this_expr))
      return true;
  }

  pool =
    member_exprt(cur_this_expr, "$dynamic_pool", symbol_typet(lib_prefix + "BytesPool"));

  return false;
}

bool solidity_convertert::get_dynamic_pool(
  const nlohmann::json &expr,
  exprt &pool)
{
  std::string c_name;
  get_current_contract_name(expr, c_name);
  if (c_name.empty())
    return true;
  return get_dynamic_pool(c_name, pool);
}

/**
 * @brief Recursively checks whether a Solidity AST node contains
 *        any usage of array-specific operations: push, pop, or length,
 *        excluding operations on the `bytes` type.
 *
 * This function traverses the AST (produced by solc in JSON format)
 * and looks for `MemberAccess` nodes with member names "push", "pop",
 * or "length". It ensures that these accesses are on array types
 * (e.g., `int[]`, `uint256[]`) and not on the dynamic `bytes` type.
 *
 * @param node A JSON node from the Solidity AST.
 * @return true if any array push/pop/length usage is found (excluding bytes).
 * @return false otherwise.
 */
bool solidity_convertert::check_array_push_pop_length(
  const nlohmann::json &node)
{
  auto is_array_type_not_bytes = [](const nlohmann::json &type_desc) -> bool {
    if (!type_desc.is_object())
      return false;
    if (!type_desc.contains("typeString"))
      return false;

    std::string type_str = type_desc["typeString"];
    if (type_str == "bytes")
      return false;
    if (type_str.find("[]") != std::string::npos)
      return true;
    return false;
  };

  if (node.is_object())
  {
    if (node.contains("nodeType") && node["nodeType"] == "MemberAccess")
    {
      if (
        node.contains("memberName") &&
        (node["memberName"] == "push" || node["memberName"] == "pop" ||
         node["memberName"] == "length"))
      {
        if (
          node.contains("expression") &&
          node["expression"].contains("typeDescriptions") &&
          is_array_type_not_bytes(node["expression"]["typeDescriptions"]))
        {
          return true;
        }
      }
    }

    for (const auto &kv : node.items())
    {
      if (check_array_push_pop_length(kv.value()))
        return true;
    }
  }
  else if (node.is_array())
  {
    for (const auto &element : node)
    {
      if (check_array_push_pop_length(element))
        return true;
    }
  }

  return false;
}

namespace
{
// Walk an expression subtree to see whether it ultimately references the
// state-var with the given declaration id.  Returns true for direct
// `Identifier(referencedDeclaration=id)` and indirectly through any chain
// of `IndexAccess.baseExpression` / `MemberAccess.expression`.  Other node
// shapes (function-call results, ternary operators, etc.) return false —
// the gate caller's job is to be conservative; an unknown source is treated
// as "not pointing to this mapping".
bool expr_targets_decl(const nlohmann::json &expr, int decl_id)
{
  if (!expr.is_object())
    return false;
  const std::string nt = expr.value("nodeType", std::string());
  if (nt == "Identifier")
    return expr.value("referencedDeclaration", -1) == decl_id;
  if (nt == "IndexAccess" && expr.contains("baseExpression"))
    return expr_targets_decl(expr["baseExpression"], decl_id);
  if (nt == "MemberAccess" && expr.contains("expression"))
    return expr_targets_decl(expr["expression"], decl_id);
  if (nt == "TupleExpression" && expr.contains("components"))
  {
    for (const auto &c : expr["components"])
      if (expr_targets_decl(c, decl_id))
        return true;
  }
  return false;
}

// Recursively scan for a `VariableDeclarationStatement` whose declared
// variable has `storageLocation == "storage"` AND whose initializer
// references `mapping_decl_id`.  Mirrors the storage-ref alias creation
// in solidity_convert_decl.cpp:388-411.
bool walk_for_storage_ref(const nlohmann::json &node, int mapping_decl_id)
{
  if (node.is_object())
  {
    const std::string nt = node.value("nodeType", std::string());
    // VariableDeclaration with storageLocation == "storage"
    if (nt == "VariableDeclaration" &&
        node.value("storageLocation", std::string()) == "storage")
    {
      // Only the *initializer* matters for alias detection.  The decl
      // itself just declares a storage pointer; the alias is established
      // on assignment.  Initializer lives on parent VariableDeclarationStatement
      // — we'll catch it when we visit that node, below.
    }
    if (nt == "VariableDeclarationStatement" &&
        node.contains("declarations") &&
        node.contains("initialValue"))
    {
      bool any_storage = false;
      for (const auto &d : node["declarations"])
      {
        if (d.is_object() &&
            d.value("storageLocation", std::string()) == "storage")
        {
          any_storage = true;
          break;
        }
      }
      if (any_storage &&
          expr_targets_decl(node["initialValue"], mapping_decl_id))
        return true;
    }
    // Direct assignment to an existing storage pointer:
    //   `T[N] storage p = m[k];` is the decl form above, but
    //   `p = m[k];` after-the-fact also creates an alias.
    if (nt == "Assignment" && node.contains("leftHandSide") &&
        node.contains("rightHandSide"))
    {
      // Check if LHS type is "storage ref" by inspecting typeString.
      const auto &lhs = node["leftHandSide"];
      if (lhs.is_object() && lhs.contains("typeDescriptions") &&
          lhs["typeDescriptions"].is_object() &&
          lhs["typeDescriptions"].value("typeString", std::string())
              .find(" storage ref") != std::string::npos &&
          expr_targets_decl(node["rightHandSide"], mapping_decl_id))
        return true;
    }
    for (const auto &kv : node.items())
      if (walk_for_storage_ref(kv.value(), mapping_decl_id))
        return true;
  }
  else if (node.is_array())
  {
    for (const auto &el : node)
      if (walk_for_storage_ref(el, mapping_decl_id))
        return true;
  }
  return false;
}

// Count the number of trailing IndexAccess nodes on top of a given
// Identifier(decl_id).  e.g. for `m[k][i]`, depth = 2 from the Identifier
// up; for `m[k]`, depth = 1; for bare `m`, depth = 0.  The walker sees the
// outermost expression first; if the outer IndexAccess's baseExpression
// chain ends at our mapping Identifier, the depth equals the chain length.
//
// Returns the maximum depth seen across all references in the subtree.
// 0 means the mapping is referenced WITHOUT any IndexAccess wrapping
// (passed by name, e.g. `f(m)` — partial access for our purposes).
unsigned max_indexaccess_depth_to(const nlohmann::json &node,
                                  int mapping_decl_id,
                                  unsigned current_depth = 0)
{
  if (!node.is_object() && !node.is_array())
    return 0;

  unsigned best = 0;

  if (node.is_object())
  {
    const std::string nt = node.value("nodeType", std::string());

    // Direct Identifier: if it matches our mapping, its depth in the
    // enclosing access chain is current_depth.
    if (nt == "Identifier" &&
        node.value("referencedDeclaration", -1) == mapping_decl_id)
      return current_depth;

    // IndexAccess: descend into baseExpression with depth+1, but ALSO
    // descend into indexExpression with depth=0 (a fresh subtree).
    if (nt == "IndexAccess" && node.contains("baseExpression"))
    {
      best = std::max(best,
        max_indexaccess_depth_to(
          node["baseExpression"], mapping_decl_id, current_depth + 1));
      if (node.contains("indexExpression"))
        best = std::max(best,
          max_indexaccess_depth_to(
            node["indexExpression"], mapping_decl_id, 0));
      return best;
    }

    // Any other node: descend into all subtrees with fresh depth=0.
    for (const auto &kv : node.items())
      best = std::max(best,
        max_indexaccess_depth_to(kv.value(), mapping_decl_id, 0));
  }
  else
  {
    for (const auto &el : node)
      best = std::max(best,
        max_indexaccess_depth_to(el, mapping_decl_id, 0));
  }
  return best;
}

// Return true if the subtree contains any reference to the mapping that
// is NOT inside an IndexAccess chain at least `expected_depth` deep.
// Used to detect partial accesses (`m`, `m[k]` for 2D, etc.).
bool walk_for_partial_access(const nlohmann::json &node,
                             int mapping_decl_id,
                             unsigned expected_depth)
{
  if (node.is_object())
  {
    const std::string nt = node.value("nodeType", std::string());

    // At an IndexAccess, check if its chain ends at our mapping.  If yes,
    // count the chain depth — depth less than expected is a partial access.
    if (nt == "IndexAccess" && node.contains("baseExpression"))
    {
      unsigned d = max_indexaccess_depth_to(
        node, mapping_decl_id, /*current_depth=*/0);
      if (d > 0 && d < expected_depth)
        return true;
      // Even if this chain is OK (d == expected_depth or d == 0), we still
      // need to recurse into the indexExpression subtree.  The full-depth
      // chain is "consumed" — don't recurse the baseExpression further.
      if (d == expected_depth && node.contains("indexExpression"))
        return walk_for_partial_access(
          node["indexExpression"], mapping_decl_id, expected_depth);
      // Otherwise (d == 0 — chain doesn't touch our mapping), recurse all kids.
    }

    // Direct Identifier match outside any IndexAccess chain: depth 0,
    // partial access.
    if (nt == "Identifier" &&
        node.value("referencedDeclaration", -1) == mapping_decl_id)
      return true;

    for (const auto &kv : node.items())
      if (walk_for_partial_access(kv.value(), mapping_decl_id, expected_depth))
        return true;
  }
  else if (node.is_array())
  {
    for (const auto &el : node)
      if (walk_for_partial_access(el, mapping_decl_id, expected_depth))
        return true;
  }
  return false;
}
} // namespace

bool solidity_convertert::has_mapping_storage_ref(
  int mapping_decl_id,
  const std::string &contract_name) const
{
  // Scan ALL contract definitions in the source unit (inheritance,
  // libraries, helper contracts can all access an inherited mapping).
  // Conservative: a single hit anywhere blocks flat encoding.
  (void)contract_name; // unused — we scan whole-source to be safe
  if (!src_ast_json.contains("nodes"))
    return false;
  for (const auto &top : src_ast_json["nodes"])
  {
    if (!top.is_object())
      continue;
    if (top.value("nodeType", std::string()) != "ContractDefinition")
      continue;
    if (walk_for_storage_ref(top, mapping_decl_id))
      return true;
  }
  return false;
}

bool solidity_convertert::has_partial_mapping_access(
  int mapping_decl_id,
  unsigned expected_access_depth,
  const std::string &contract_name) const
{
  (void)contract_name; // unused — scan whole-source
  if (!src_ast_json.contains("nodes"))
    return false;
  for (const auto &top : src_ast_json["nodes"])
  {
    if (!top.is_object())
      continue;
    if (top.value("nodeType", std::string()) != "ContractDefinition")
      continue;
    if (walk_for_partial_access(top, mapping_decl_id, expected_access_depth))
      return true;
  }
  return false;
}

unsigned solidity_convertert::array_nesting_depth(
  const typet &fixed_array_t) const
{
  unsigned depth = 0;
  const typet *cur = &fixed_array_t;
  while (cur->is_array())
  {
    const exprt &sz = to_array_type(*cur).size();
    if (sz.is_nil() || sz.id() == "infinity")
      break; // stop at a dynamic boundary
    ++depth;
    cur = &cur->subtype();
  }
  return depth;
}

bool solidity_convertert::compute_flat_extent(
  const typet &fixed_array_t,
  unsigned long &inner_extent,
  typet &leaf_t) const
{
  inner_extent = 1;
  const typet *cur = &fixed_array_t;
  while (cur->is_array())
  {
    const exprt &sz = to_array_type(*cur).size();
    if (sz.is_nil() || sz.id() == "infinity")
      return false; // dynamic-size — not eligible
    BigInt n;
    if (to_integer(sz, n))
      return false; // non-constant size
    if (n.is_negative() || !n.is_uint64())
      return false;
    unsigned long ni = n.to_uint64();
    // overflow guard: keep extent < 2^32 so the bit-shift fits comfortably
    if (ni == 0 || ni > (1ul << 32) || inner_extent > (1ul << 32) ||
        inner_extent * ni > (1ul << 32))
      return false;
    inner_extent *= ni;
    cur = &cur->subtype();
  }
  // The chain terminator must be a non-array leaf.
  if (cur->is_array())
    return false;
  leaf_t = *cur;
  return true;
}

// check if current contract have bytes (not bytesN) type
bool solidity_convertert::has_contract_bytes(const nlohmann::json &node)
{
  if (node.is_object())
  {
    // 0.6.x ElementaryTypeName nodes carry an inner typeDescriptions whose
    // typeString is JSON null; guard against the null before converting.
    if (
      node.contains("typeDescriptions") &&
      node["typeDescriptions"].is_object() &&
      node["typeDescriptions"].contains("typeString") &&
      node["typeDescriptions"]["typeString"].is_string())
    {
      const std::string &ts = node["typeDescriptions"]["typeString"].get_ref<const std::string &>();
      // Match "bytes", "bytes storage pointer", "bytes memory", etc.
      // Also match "string" variants since string uses BytesDynamic internally.
      // Inline literals carry the `literal_string "..."` / `literal_bytes
      // "..."` typeString prefix — they decay to `bytes`/`string` at the
      // call site and therefore drive `$dynamic_pool` use just like a
      // declared state var would.
      if (
        ts == "bytes" || ts.substr(0, 6) == "bytes " ||
        ts == "string" || ts.substr(0, 7) == "string " ||
        ts.substr(0, 14) == "literal_string" ||
        ts.substr(0, 13) == "literal_bytes")
        return true;
    }

    for (const auto &kv : node.items())
    {
      if (has_contract_bytes(kv.value()))
        return true;
    }
  }
  else if (node.is_array())
  {
    for (const auto &element : node)
    {
      if (has_contract_bytes(element))
        return true;
    }
  }

  return false;
}

void solidity_convertert::gen_mapping_key_typecast(
  const std::string &c_name,
  exprt &pos,
  const locationt &location,
  const typet &key_type)
{
  SolidityGrammar::SolType key_sol_type = get_sol_type(key_type);
  if (
    key_sol_type == SolidityGrammar::SolType::STRING ||
    key_sol_type == SolidityGrammar::SolType::STRING_LITERAL)
  {
    side_effect_expr_function_callt str2uint_call;
    assert(context.find_symbol("c:@F@str2uint") != nullptr);
    get_library_function_call_no_args(
      "str2uint",
      "c:@F@str2uint",
      unsignedbv_typet(256),
      location,
      str2uint_call);
    str2uint_call.arguments().push_back(pos);
    pos = str2uint_call;
    solidity_gen_typecast(ns, pos, unsignedbv_typet(256));
    return;
  }
  // bytesN: use bytes_static_to_mapping_key
  else if (is_bytesN_type(key_type))
  {
    // bytes_static_to_mapping_key(pos)
    side_effect_expr_function_callt call;
    get_library_function_call_no_args(
      "bytes_static_to_mapping_key",
      "c:@F@bytes_static_to_mapping_key",
      unsignedbv_typet(256),
      location,
      call);
    call.arguments().push_back(pos);
    pos = call;
    return;
  }
  else if (is_bytes_type(key_type))
  {
    side_effect_expr_function_callt bytes_dynamic_call;
    assert(context.find_symbol("c:@F@bytes_dynamic_to_mapping_key") != nullptr);
    get_library_function_call_no_args(
      "bytes_dynamic_to_mapping_key",
      "c:@F@bytes_dynamic_to_mapping_key",
      unsignedbv_typet(256),
      location,
      bytes_dynamic_call);
    bytes_dynamic_call.arguments().push_back(pos);

    // get dynamic_pool from current contract instance
    // get this
    exprt dynamic_pool_member;
    if (get_dynamic_pool(c_name, dynamic_pool_member))
      abort();

    bytes_dynamic_call.arguments().push_back(dynamic_pool_member);

    pos = bytes_dynamic_call;
    return;
  }
  // fallback for all others: keep old logic
  solidity_gen_typecast(ns, pos, unsignedbv_typet(256));
}

void solidity_convertert::combine_mapping_keys_256(
  const std::vector<exprt> &folded_keys_64,
  exprt &combined)
{
  // Pack folded 64-bit keys into successive 64-bit lanes of a uint256.
  // Lane assignment for lane i = i mod 4 (so lanes 0..3 are unique, then
  // lanes 4+ XOR-collide back into lane 0..3).  Lane mod-4 collision is
  // acceptable for nesting depths > 4: it merely conflates distinct
  // chains, which becomes a TOD false positive — never a crash, never
  // a missed write/read pairing for fixed depth ≤ 4.
  const typet u256 = unsignedbv_typet(256);
  combined = from_integer(0, u256);
  for (size_t i = 0; i < folded_keys_64.size(); ++i)
  {
    exprt k = folded_keys_64[i];
    solidity_gen_typecast(ns, k, u256);
    if ((i % 4) > 0)
    {
      exprt shift = from_integer(64 * (i % 4), u256);
      exprt shifted("shl", u256);
      shifted.copy_to_operands(k, shift);
      k = shifted;
    }
    exprt or_expr("bitor", u256);
    or_expr.copy_to_operands(combined, k);
    combined = or_expr;
  }
}

void solidity_convertert::xor_fold_key_to_64bit(exprt &key)
{
  // 2026-05-01: function name kept for callsite compatibility but
  // body is now a 256-bit identity normalise. The wide-BV array index
  // infrastructure (commit 9bd1cecf92) lets per-mapping array_typets
  // carry an explicit index_width = 256 so the SMT layer can index
  // wider than word_size; the per-decl annotations are added in
  // solidity_convert_decl.cpp at array creation time. Closes
  // ledger #22's 256→64 fold unsoundness for path-1 (frontend
  // direct index_exprt) accesses.

  const typet u256 = unsignedbv_typet(256);
  // Normalise the key to a plain 256-bit value. For bytes32 mapping
  // keys the frontend models them as the BytesStatic struct (array +
  // length), and `shr(BytesStatic, uint256)` has no irep2 mapping —
  // casting through solidity_gen_typecast routes through the
  // bytesN→uint256 lowering.
  if (!key.type().is_unsignedbv())
    solidity_gen_typecast(ns, key, u256);
  // Coerce narrower unsignedbv (e.g. address_t = uint160) to uint256.
  // For wider unsigned types (uint256), the cast is a no-op.
  if (key.type() != u256)
    solidity_gen_typecast(ns, key, u256);
}

/**
  index accesss could either be set or get:
  x[1]      => map_uint_get(&m, 1)
  x[1] = 2  => map_uint_set(&x, 1, 2)
  @array: x
  @pos: 1
  @is_mapping_set: true if it's a setValue, otherwise getValue
*/
bool solidity_convertert::get_new_mapping_index_access(
  const typet &value_t,
  SolidityGrammar::SolType val_sol_type,
  bool is_mapping_set,
  const exprt &array,
  const exprt &pos,
  const locationt &location,
  exprt &new_expr)
{
  std::string val_flg;
  typet func_type;
  if (
    SolidityGrammar::is_uint_type(val_sol_type) ||
    SolidityGrammar::is_bytes_type(val_sol_type) ||
    SolidityGrammar::is_address_type(val_sol_type) ||
    val_sol_type == SolidityGrammar::SolType::ENUM)
  {
    val_flg = "uint";
    func_type = unsignedbv_typet(256);
  }
  else if (SolidityGrammar::is_int_type(val_sol_type))
  {
    val_flg = "int";
    func_type = signedbv_typet(256);
  }
  else if (val_sol_type == SolidityGrammar::SolType::BOOL)
  {
    val_flg = "bool";
    func_type = bool_typet();
  }
  else if (
    val_sol_type == SolidityGrammar::SolType::STRING ||
    val_sol_type == SolidityGrammar::SolType::STRING_LITERAL)
  {
    val_flg = "string";
    func_type = value_t;
  }
  else if (val_sol_type == SolidityGrammar::SolType::DYNARRAY)
  {
    /* Dedicated dynarray dispatch (2026-04-21): the mapping stores a
     * pointer-to-pointer so push writeback can rewrite the slot after
     * `_ESBMC_array_push_uint256` allocates a new slab. Cannot reuse
     * `map_generic_set/get` — those copy the value by sizeof, but for a
     * dynarray the "value" IS the heap pointer, and successive pushes
     * need to update that pointer in place. */
    val_flg = "dynarr";
    func_type = pointer_typet(empty_typet());
  }
  else if (
    val_sol_type == SolidityGrammar::SolType::ARRAY_LITERAL ||
    val_sol_type == SolidityGrammar::SolType::ARRAY)
  {
    /* Fixed-size array value (mapping(K => T[N]) / mapping(K =>
     * T[M][N])): Solidity pre-binds every key to an N-element (or
     * M*N-element) zero-filled slab. map_fixed_arr_get lazily
     * allocates that slab on first access and returns the same
     * pointer on subsequent reads, so element writes via [i]
     * (or [i][j]) persist. The 2-arg read is a special-case; writes
     * of whole T[N] slots aren't supported yet — assignment
     * `m[k] = new T[N](...)` would need a dedicated set helper.
     * 1D fixed-array value lands under ARRAY_LITERAL; 2D+ under
     * ARRAY (see solidity_convert_type.cpp:360 vs 419). */
    val_flg = "fixed_arr";
    func_type = pointer_typet(empty_typet());
  }
  else
  {
    val_flg = "generic";
    // void *
    func_type = pointer_typet(empty_typet());
  }

  // A STRUCT value must be reached through a POINTER, not copied out by value.
  // `map_generic_get` hands back a COPY of the slot, so the frontend lowered
  //     Acc[k].b = x;   ->   aux = map_<S>_get(&Acc,k); aux.b = x;
  // i.e. the field write landed in a temporary and was LOST, and every read came
  // back as the zero default -- measured with a fixed key, so it was not a key
  // mismatch: `mapping(K=>uint)` and a plain scalar both round-trip a cross-contract
  // write, `mapping(K=>Struct)` did not. `map_fixed_arr_get` is the existing
  // lazily-allocating, pointer-returning helper (it treats the payload as an opaque
  // byte slab, so it is value-type agnostic); routing STRUCT through it makes reads
  // and writes hit the SAME slot.
  const bool struct_by_ref =
    (val_flg == "generic" && val_sol_type == SolidityGrammar::SolType::STRUCT);
  if (struct_by_ref)
    val_flg = "fixed_arr";

  // construct func call
  std::string func_name;
  if (is_mapping_set)
  {
    func_name = "map_" + val_flg + "_set";
    func_type = empty_typet();
    // overwrite func_type
    func_type.set("cpp_type", "void");
  }
  else
    func_name = "map_" + val_flg + "_get";

  if (context.find_symbol("c:@F@" + func_name) == nullptr)
  {
    log_error(
      "cannot find mapping ref {}. Got val_sol_type={}",
      func_name,
      SolidityGrammar::sol_type_to_str(val_sol_type));
    return true;
  }
  side_effect_expr_function_callt call;
  get_library_function_call_no_args(
    func_name, "c:@F@" + func_name, func_type, location, call);

  // &array
  call.arguments().push_back(address_of_exprt(array));

  // index
  call.arguments().push_back(pos);

  // Fixed-size array get: append sizeof(T[N]) so the helper can lazily
  // calloc the right amount on first access. Set path is rejected above
  // (no fixed_arr_set helper emitted); this branch is read-only.
  if (val_flg == "fixed_arr" && !is_mapping_set)
  {
    exprt size_of_expr;
    get_size_of_expr(value_t, size_of_expr);
    call.arguments().push_back(size_of_expr);
  }

  if (is_mapping_set)
  {
    /*
        case 1: x[1] += 2 =>
          DECL temp = map_uint_get(&array, pos);  <-- move to front block
          temp += 2;
          map_uint_set(&array, pos, temp);  <-- move to back block
          (map_generic_set(&array, pos, temp, sizeof(temp));)
    */
    std::string aux_name, aux_id;
    get_aux_var(aux_name, aux_id);
    symbolt aux_sym;
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    typet aux_type = value_t;
    get_default_symbol(
      aux_sym, debug_modulename, aux_type, aux_name, aux_id, location);
    aux_sym.file_local = true;
    aux_sym.lvalue = true;
    auto &added_sym = *move_symbol_to_context(aux_sym);
    code_declt decl(symbol_expr(added_sym));

    // populate initial value
    side_effect_expr_function_callt get_call;
    std::string f_get_name = "map_" + val_flg + "_get";

    get_library_function_call_no_args(
      f_get_name, "c:@F@" + f_get_name, value_t, location, get_call);
    get_call.arguments().push_back(address_of_exprt(array));
    get_call.arguments().push_back(pos);
    solidity_gen_typecast(ns, get_call, aux_type);
    added_sym.value = get_call;
    decl.operands().push_back(get_call);
    move_to_front_block(decl);

    // value
    call.arguments().push_back(symbol_expr(added_sym));
    if (val_flg == "generic")
    {
      // sizeof
      exprt size_of_expr;
      get_size_of_expr(value_t, size_of_expr);
      call.arguments().push_back(size_of_expr);
    }

    convert_expression_to_code(call);
    move_to_back_block(call);

    new_expr = symbol_expr(added_sym);
  }
  else if (val_flg == "generic")
  {
    // The "generic" val_flg covers every value type that isn't a
    // scalar (uint/int/bool/string).  Each of these requires a
    // different access shape.  Dispatch explicitly by val_sol_type so
    // we don't fall through to the STRUCT path on e.g. a dynamic
    // array whose `value_t.identifier()` is empty — that would crash
    // in `get_mapping_struct_function` at `substr(prefix.length())`.
    //
    //  - MAPPING  (nested `mapping(K1=>mapping(K2=>V))`): hand back
    //    the raw `map_generic_get(void*)` call.  The caller
    //    (`get_index_access_expr`) adds another `index_exprt`.
    //  - DYNARRAY (`mapping(K => T[])`): same shape — return the raw
    //    void* call.  Downstream `.push`/`.pop`/`[i]` lowerings
    //    treat it as a pointer to the stored array's data.
    //  - STRUCT   (`mapping(K => S)`): go through
    //    `map_<Struct>_get` so the caller can access fields.
    if (
      val_sol_type == SolidityGrammar::SolType::MAPPING ||
      val_sol_type == SolidityGrammar::SolType::DYNARRAY)
    {
      new_expr = call;
      return false;
    }

    if (val_sol_type != SolidityGrammar::SolType::STRUCT)
    {
      log_error(
        "unsupported mapping value type: sol_type={}",
        SolidityGrammar::sol_type_to_str(val_sol_type));
      return true;
    }

    /* generic_get:
          case 2: users[msg.sender].age; =>
            DECL struct temp = map_users_get(&array, pos);
            temp.age;
        */
    std::string aux_name, aux_id;
    get_aux_var(aux_name, aux_id);
    symbolt aux_sym;
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    typet aux_type = value_t; // struct *
    get_default_symbol(
      aux_sym, debug_modulename, aux_type, aux_name, aux_id, location);
    aux_sym.file_local = true;
    aux_sym.lvalue = true;
    auto &added_sym = *move_symbol_to_context(aux_sym);
    code_declt decl(symbol_expr(added_sym));

    // construct map_{struct_name}_get() function
    // e.g. map_Base_User_get();
    exprt map_struct_get;
    std::string struct_contract_name = value_t.identifier().as_string();
    if (struct_contract_name.empty())
    {
      log_error(
        "mapping value type has empty identifier: sol_type={}",
        SolidityGrammar::sol_type_to_str(val_sol_type));
      return true;
    }
    get_mapping_struct_function(
      value_t, struct_contract_name, call, map_struct_get);

    // struct temp = map_users_get(&array, pos);
    added_sym.value = map_struct_get;
    decl.operands().push_back(map_struct_get);
    move_to_front_block(decl);

    new_expr = symbol_expr(added_sym);
  }
  else if (struct_by_ref)
  {
    // (struct S *)map_fixed_arr_get(&m, k, sizeof(S)) dereferenced, so the caller's
    // `.field` access is an LVALUE into the stored slot.
    typet sptr = gen_pointer_type(value_t);
    solidity_gen_typecast(ns, call, sptr);
    new_expr = dereference_exprt(call, value_t);
    new_expr.location() = location;
  }
  else if (val_flg == "fixed_arr")
  {
    /* map_fixed_arr_get returns a void* to the N-element slab. Cast
     * to `pointer<element>` so the downstream `[i]` index lowering
     * uses the correct element size in pointer arithmetic — without
     * the cast, `index_exprt(void*, i, T)` strides by sizeof(void)
     * (== 1) on the GOTO side, producing a byte-level read instead
     * of an element read. Do NOT cast to value_t (the array type
     * itself) — solidity_gen_typecast rejects array destinations.
     * value_t is `T[N]`; its subtype is the element type `T`. */
    typet elem_ptr_t = pointer_typet(value_t.subtype());
    solidity_gen_typecast(ns, call, elem_ptr_t);
    new_expr = call;
  }
  else
  {
    // e.g. (int8)map_int_get(&arr, 1);
    solidity_gen_typecast(ns, call, value_t);
    new_expr = call;
  }

  return false;
}

void solidity_convertert::get_mapping_struct_function(
  const typet &struct_t,
  std::string &struct_contract_name,
  const side_effect_expr_function_callt &gen_call,
  exprt &new_expr)
{
  /*
  e.g.
  struct A map_get_A_default_val(struct mapping_t *m, uint256_t k)
  {
  __ESBMC_HIDE:;
    struct A *ap = (struct A *)map_get_generic(m, k);
    return ap ? *ap : (struct A){0};
  }
  */
  side_effect_expr_function_callt call;

  // split contract struct name
  // drop prefix
  struct_contract_name = struct_contract_name.substr(prefix.length());
  // replace "." to "_"
  std::replace(
    struct_contract_name.begin(), struct_contract_name.end(), '.', '_');
  std::string func_name = "map_" + struct_contract_name + "_get";
  std::string func_id = "c:@F@" + func_name;
  if (context.find_symbol(func_id) != nullptr)
  {
    call.function() = symbol_expr(*context.find_symbol(func_id));
    call.type() = struct_t;
    for (auto &arg : gen_call.arguments())
      call.arguments().push_back(arg); // same arugments as map_get_generic
    new_expr = call;
    return;
  }

  std::string debug_modulename = get_modulename_from_path(absolute_path);
  code_typet func_t;
  func_t.return_type() = struct_t;

  // PARAMETERS (the doc comment above always described this shape:
  //   struct A map_A_get(struct mapping_t *m, uint256_t k)
  // ) -- previously the helper was declared with NO arguments and its body
  // embedded `gen_call`, i.e. the FIRST call site's argument EXPRESSIONS
  // (`&this->c->Acc`, `this->$address`).  Inside a free function those have no
  // `this`, so the body was unresolvable and symex pruned every path through a
  // `mapping(K => Struct)` public getter (measured: `assert(false)` right after
  // such a read is UNREACHABLE, while the same probe on `mapping(K => uint)`
  // reports it FAILED).  The helper is also CACHED per struct name, so a second
  // call site would have silently read the first site's mapping.  Declare real
  // parameters and build the body from them.
  std::vector<std::string> param_ids;
  param_ids.reserve(gen_call.arguments().size());
  for (size_t i = 0; i < gen_call.arguments().size(); ++i)
  {
    std::string pname = "_arg" + std::to_string(i);
    std::string pid = func_id + "@" + pname;
    symbolt ps;
    get_default_symbol(
      ps,
      debug_modulename,
      gen_call.arguments()[i].type(),
      pname,
      pid,
      gen_call.location());
    ps.lvalue = true;
    ps.is_parameter = true;
    ps.file_local = true;
    move_symbol_to_context(ps);

    code_typet::argumentt param;
    param.type() = gen_call.arguments()[i].type();
    param.cmt_base_name(pname);
    param.cmt_identifier(pid);
    func_t.arguments().push_back(param);
    param_ids.push_back(pid);
  }

  symbolt sym;
  get_default_symbol(
    sym, debug_modulename, func_t, func_name, func_id, gen_call.location());
  sym.file_local = true;
  auto &func_sym = *move_symbol_to_context(sym);

  code_blockt func_body;
  // hide it
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.move_to_operands(label);

  // struct A *ap = (struct A *)map_get_generic(m, k);
  std::string aux_name, aux_id;
  get_aux_var(aux_name, aux_id);
  symbolt aux_sym;
  typet aux_type = gen_pointer_type(struct_t); // struct *
  get_default_symbol(
    aux_sym, debug_modulename, aux_type, aux_name, aux_id, gen_call.location());
  aux_sym.file_local = true;
  aux_sym.lvalue = true;
  auto &added_sym = *move_symbol_to_context(aux_sym);
  code_declt decl(symbol_expr(added_sym));
  // for typcast -- call map_generic_get on the PARAMETERS, not on the call
  // site's expressions (see the parameter block above).
  side_effect_expr_function_callt temp_call = gen_call;
  for (size_t i = 0; i < temp_call.arguments().size() && i < param_ids.size();
       ++i)
  {
    const symbolt *ps = context.find_symbol(param_ids[i]);
    assert(ps != nullptr);
    temp_call.arguments()[i] = symbol_expr(*ps);
  }
  solidity_gen_typecast(ns, temp_call, aux_type);
  added_sym.value = temp_call;
  decl.operands().push_back(temp_call);
  // move to func body
  func_body.operands().push_back(decl);

  // ternary: return ap ? *ap : (struct A){0};
  // we split it into
  // - struct A aux = {0};
  // - return ap ? *ap : aux;

  // construct empty struct instance
  std::string aux_name2, aux_id2;
  get_aux_var(aux_name2, aux_id2);
  symbolt aux_sym2;
  typet aux_type2 = struct_t; // struct *
  get_default_symbol(
    aux_sym2,
    debug_modulename,
    aux_type2,
    aux_name2,
    aux_id2,
    gen_call.location());
  aux_sym2.file_local = true;
  aux_sym2.lvalue = true;
  auto &added_sym2 = *move_symbol_to_context(aux_sym2);
  code_declt decl2(symbol_expr(added_sym2));
  // zero value
  exprt inits = gen_zero(get_complete_type(aux_type2, ns), true);
  added_sym2.value = inits;
  decl2.operands().push_back(inits);
  // move to func body
  func_body.operands().push_back(decl2);

  // ap ? *ap : aux;
  exprt if_expr("if", struct_t);
  if_expr.operands().push_back(symbol_expr(added_sym));
  if_expr.operands().push_back(
    dereference_exprt(symbol_expr(added_sym), struct_t));
  if_expr.operands().push_back(symbol_expr(added_sym2));
  if_expr.location() = gen_call.location();

  // return ap ? *ap : aux;
  code_returnt ret;
  ret.return_value() = if_expr;
  func_body.operands().push_back(ret);

  func_sym.value = func_body;

  // func call
  call.function() = symbol_expr(func_sym);
  call.type() = struct_t;
  for (auto &arg : gen_call.arguments())
    call.arguments().push_back(arg); // same arugments as map_get_generic
  new_expr = call;
}

// invoking a function in the library
// note that the function symbol might not be inside the symbol table at the moment
void solidity_convertert::get_library_function_call_no_args(
  const std::string &func_name,
  const std::string &func_id,
  const typet &t,
  const locationt &l,
  exprt &new_expr)
{
  side_effect_expr_function_callt call_expr;

  exprt type_expr("symbol");
  type_expr.name(func_name);
  type_expr.pretty_name(func_name);
  type_expr.identifier(func_id);

  typet type;
  if (t.is_code())
    // this means it's a func symbol read from the symbol_table
    type = to_code_type(t).return_type();
  else
    type = t;

  call_expr.function() = type_expr;
  call_expr.type() = type;

  call_expr.location() = l;
  new_expr = call_expr;
}

void solidity_convertert::get_malloc_function_call(
  const locationt &loc,
  side_effect_expr_function_callt &malc_call)
{
  const std::string malc_name = "malloc";
  const std::string malc_id = "c:@F@malloc";
  const symbolt &malc_sym = *context.find_symbol(malc_id);
  get_library_function_call_no_args(
    malc_name, malc_id, symbol_expr(malc_sym).type(), loc, malc_call);
}

void solidity_convertert::get_calloc_function_call(
  const locationt &loc,
  side_effect_expr_function_callt &calc_call)
{
  const std::string calc_name = "_ESBMC_alloc_array";
  const std::string calc_id = "c:@F@_ESBMC_alloc_array";
  const symbolt &calc_sym = *context.find_symbol(calc_id);
  get_library_function_call_no_args(
    calc_name, calc_id, symbol_expr(calc_sym).type(), loc, calc_call);
}

void solidity_convertert::get_malloc_array_function_call(
  const locationt &loc,
  side_effect_expr_function_callt &malc_call)
{
  const std::string malc_name = "_ESBMC_alloc_array_sym";
  const std::string malc_id = "c:@F@_ESBMC_alloc_array_sym";
  const symbolt &malc_sym = *context.find_symbol(malc_id);
  get_library_function_call_no_args(
    malc_name, malc_id, symbol_expr(malc_sym).type(), loc, malc_call);
}

void solidity_convertert::get_arrcpy_function_call(
  const locationt &loc,
  side_effect_expr_function_callt &calc_call)
{
  const std::string calc_name = "_ESBMC_arrcpy";
  const std::string calc_id = "c:@F@_ESBMC_arrcpy";
  const symbolt &calc_sym = *context.find_symbol(calc_id);
  get_library_function_call_no_args(
    calc_name, calc_id, symbol_expr(calc_sym).type(), loc, calc_call);
}

void solidity_convertert::get_arrcpy_2d_function_call(
  const locationt &loc,
  side_effect_expr_function_callt &calc_call)
{
  const std::string calc_name = "_ESBMC_arrcpy_2d";
  const std::string calc_id = "c:@F@_ESBMC_arrcpy_2d";
  const symbolt &calc_sym = *context.find_symbol(calc_id);
  get_library_function_call_no_args(
    calc_name, calc_id, symbol_expr(calc_sym).type(), loc, calc_call);
}

void solidity_convertert::get_str_assign_function_call(
  const locationt &loc,
  side_effect_expr_function_callt &_call)
{
  const std::string func_name = "_str_assign";
  const std::string func_id = "c:@F@_str_assign";
  const symbolt &func_sym = *context.find_symbol(func_id);
  get_library_function_call_no_args(
    func_name, func_id, symbol_expr(func_sym).type(), loc, _call);
}

void solidity_convertert::get_memcpy_function_call(
  const locationt &loc,
  side_effect_expr_function_callt &memc_call)
{
  const std::string memc_name = "memcpy";
  const std::string memc_id = "c:@F@memcpy";
  const symbolt &memc_sym = *context.find_symbol(memc_id);
  get_library_function_call_no_args(
    memc_name, memc_id, symbol_expr(memc_sym).type(), loc, memc_call);
}

// check if the function is a library function (defined in solidity.h)
bool solidity_convertert::is_esbmc_library_function(const std::string &id)
{
  if (context.find_symbol(id) == nullptr)
    return false;
  if (id.compare(0, 3, "c:@") == 0)
    return true;
  return false;
}
