#include <solidity-frontend/solidity_tod_harness.h>
#include <solidity-frontend/solidity_tod_analysis.h>
#include <util/message/message.h>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <regex>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a Solidity AST "src" field ("offset:length:fileIndex") and
/// return the substring of the original source.
static std::string extract_src(
  const std::string &source,
  const std::string &src_field)
{
  auto colon1 = src_field.find(':');
  auto colon2 = src_field.find(':', colon1 + 1);
  if (colon1 == std::string::npos || colon2 == std::string::npos)
    return {};
  size_t offset = std::stoull(src_field.substr(0, colon1));
  size_t length = std::stoull(src_field.substr(colon1 + 1, colon2 - colon1 - 1));
  if (offset + length > source.size())
    return {};
  return source.substr(offset, length);
}

/// Find a ContractDefinition node by name in the top-level AST.
static const nlohmann::json *find_contract(
  const nlohmann::json &ast,
  const std::string &name)
{
  if (!ast.contains("nodes"))
    return nullptr;
  for (const auto &node : ast["nodes"])
  {
    if (
      node.value("nodeType", "") == "ContractDefinition" &&
      node.value("name", "") == name)
      return &node;
  }
  return nullptr;
}

/// Extract the names of contracts/interfaces/libraries that `node`
/// inherits from via its `baseContracts` list.  Only meaningful for
/// ContractDefinition nodes.
static std::vector<std::string> get_base_names(const nlohmann::json &node)
{
  std::vector<std::string> out;
  if (!node.contains("baseContracts") || !node["baseContracts"].is_array())
    return out;
  for (const auto &bc : node["baseContracts"])
  {
    if (!bc.contains("baseName"))
      continue;
    const auto &bn = bc["baseName"];
    // baseName is a UserDefinedTypeName or IdentifierPath; both have "name".
    if (bn.contains("name") && bn["name"].is_string())
      out.push_back(bn["name"].get<std::string>());
  }
  return out;
}

/// Collect the source text of every top-level SourceUnit-level declaration
/// (contract / interface / library / error / struct / enum / ...) that the
/// target contract transitively depends on, **in topological order** so
/// that base contracts precede any contract that inherits from them.
/// The target contract itself is EXCLUDED — its source text is emitted
/// separately by the caller.
///
/// Why "transitive dependencies only": if the target is a base of some
/// other concrete contract in the source (e.g. target = ERC20, source
/// also contains `contract Foo is ERC20`), including Foo would place Foo
/// BEFORE the target in the harness file, and Foo's `is ERC20` reference
/// would not resolve — solc errors with "Definition of base has to
/// precede definition of derived contract".  Foo is not needed for the
/// harness to exercise the target anyway.
///
/// Why "topological order": even within the transitive dependency set,
/// the SourceUnit node order may not match inheritance order after
/// ESBMC-side AST reshaping (inheritance merging / monomorphization can
/// reorder top-level nodes).  Kahn's algorithm over the inheritance DAG
/// guarantees bases-before-derived regardless of input order.
///
/// Returned strings are concatenated with \n\n separators, or empty if
/// there are no dependencies to emit.
static std::string collect_dependency_definitions(
  const nlohmann::json &ast,
  const std::string &sol_source,
  const std::string &target_name)
{
  if (!ast.contains("nodes") || !ast["nodes"].is_array())
    return {};

  // 1. Gather every eligible top-level node.
  struct Dep
  {
    const nlohmann::json *node;
    std::string name;                 // empty for non-named nodes
    std::vector<std::string> bases;   // only for contract/interface (not library)
    // `inheritable` = participates in the inheritance DAG (regular contract,
    //                 abstract contract, or interface).  Libraries and
    //                 non-contract nodes (structs, enums, free functions,
    //                 errors) never inherit and never are inherited — they
    //                 are always kept and emitted before any inherit chain.
    bool inheritable;
  };
  std::vector<Dep> deps;
  for (const auto &node : ast["nodes"])
  {
    const std::string nt = node.value("nodeType", "");
    if (nt == "PragmaDirective" || nt == "ImportDirective")
      continue;
    const std::string src_field = node.value("src", "");
    if (src_field.empty())
      continue;

    Dep d;
    d.node = &node;
    d.name = node.value("name", "");
    // ContractDefinition covers contract, abstract contract, interface,
    // and library — distinguish via contractKind.  Only the first three
    // participate in inheritance edges we care about.
    if (nt == "ContractDefinition")
    {
      const std::string kind = node.value("contractKind", "contract");
      d.inheritable = (kind != "library");
      if (d.inheritable)
        d.bases = get_base_names(node);
    }
    else
    {
      d.inheritable = false;
    }
    deps.push_back(std::move(d));
  }

  // 2. Name → index lookup for contract-to-contract edges.
  std::unordered_map<std::string, size_t> idx_of;
  for (size_t i = 0; i < deps.size(); ++i)
    if (!deps[i].name.empty())
      idx_of[deps[i].name] = i;

  // 3. Exclude contracts that descend FROM the target.  Keep everything
  //    else — the target may cast to an unrelated contract / interface
  //    (e.g. `ApproveAndCallFallBack(spender).receiveApproval(...)`),
  //    which we cannot cheaply detect from node metadata alone, so we
  //    keep those by default.  Descendants of the target, on the other
  //    hand, are guaranteed to reference the target via their `is
  //    Target` clause and would be emitted BEFORE the target declaration
  //    (which the emitter places separately afterwards), producing
  //    "Definition of base has to precede definition of derived
  //    contract" from solc.
  std::vector<bool> exclude(deps.size(), false);
  {
    // Build reverse-inheritance: for each contract, which others list it
    // as a base?
    std::vector<std::vector<size_t>> child(deps.size());
    for (size_t i = 0; i < deps.size(); ++i)
    {
      if (!deps[i].inheritable)
        continue;
      for (const auto &b : deps[i].bases)
      {
        auto it = idx_of.find(b);
        if (it != idx_of.end())
          child[it->second].push_back(i);
      }
    }
    auto target_it = idx_of.find(target_name);
    std::queue<size_t> bfs;
    if (target_it != idx_of.end())
    {
      for (size_t c : child[target_it->second])
      {
        if (!exclude[c])
        {
          exclude[c] = true;
          bfs.push(c);
        }
      }
    }
    while (!bfs.empty())
    {
      const size_t i = bfs.front();
      bfs.pop();
      for (size_t c : child[i])
      {
        if (!exclude[c])
        {
          exclude[c] = true;
          bfs.push(c);
        }
      }
    }
  }
  // Kept set: everything not excluded, and not the target itself.
  std::vector<bool> keep(deps.size(), false);
  for (size_t i = 0; i < deps.size(); ++i)
  {
    if (exclude[i])
      continue;
    if (deps[i].name == target_name)
      continue; // target is emitted separately by the caller
    keep[i] = true;
  }

  // 4. Topological sort of the `keep` set via Kahn's algorithm.  A
  //    contract's in-degree counts only bases that are also kept.
  std::vector<size_t> in_degree(deps.size(), 0);
  std::vector<std::vector<size_t>> rev(deps.size());
  for (size_t i = 0; i < deps.size(); ++i)
  {
    if (!keep[i])
      continue;
    for (const auto &b : deps[i].bases)
    {
      auto it = idx_of.find(b);
      if (it == idx_of.end() || !keep[it->second])
        continue;
      rev[it->second].push_back(i);
      in_degree[i]++;
    }
  }

  std::vector<size_t> order;
  std::queue<size_t> ready;
  for (size_t i = 0; i < deps.size(); ++i)
    if (keep[i] && in_degree[i] == 0)
      ready.push(i);
  while (!ready.empty())
  {
    const size_t i = ready.front();
    ready.pop();
    order.push_back(i);
    for (size_t j : rev[i])
      if (--in_degree[j] == 0)
        ready.push(j);
  }

  // 5. Emit kept nodes in topological order.  Any node not ordered due
  //    to a cycle (should not happen under Solidity inheritance rules)
  //    falls through to the tail in original order.
  std::ostringstream out;
  std::vector<bool> emitted(deps.size(), false);
  for (size_t i : order)
  {
    emitted[i] = true;
    std::string text =
      extract_src(sol_source, (*deps[i].node).value("src", ""));
    if (!text.empty())
      out << text << "\n\n";
  }
  for (size_t i = 0; i < deps.size(); ++i)
  {
    if (!keep[i] || emitted[i])
      continue;
    std::string text =
      extract_src(sol_source, (*deps[i].node).value("src", ""));
    if (!text.empty())
      out << text << "\n\n";
  }
  return out.str();
}

/// Find a FunctionDefinition inside a ContractDefinition by name, or
/// inside any of its linearised bases when `ast` is supplied and no
/// match is found on the target itself.  MRO order ensures the
/// leaf-most override wins when both the target and a base define a
/// same-named function.
static const nlohmann::json *find_function(
  const nlohmann::json &contract,
  const std::string &name,
  const nlohmann::json *ast = nullptr)
{
  if (contract.contains("nodes"))
  {
    for (const auto &node : contract["nodes"])
    {
      if (
        node.value("nodeType", "") == "FunctionDefinition" &&
        node.value("name", "") == name)
        return &node;
    }
  }
  if (!ast || !ast->contains("nodes") || !ast->at("nodes").is_array())
    return nullptr;
  if (
    !contract.contains("linearizedBaseContracts") ||
    !contract.at("linearizedBaseContracts").is_array())
    return nullptr;

  std::map<int, const nlohmann::json *> contract_by_id;
  for (const auto &n : (*ast)["nodes"])
  {
    if (n.value("nodeType", "") != "ContractDefinition")
      continue;
    if (!n.contains("id") || !n["id"].is_number_integer())
      continue;
    contract_by_id[n["id"].get<int>()] = &n;
  }

  const auto &lin = contract["linearizedBaseContracts"];
  for (size_t i = 1; i < lin.size(); ++i)
  {
    if (!lin[i].is_number_integer())
      continue;
    int bid = lin[i].get<int>();
    auto it = contract_by_id.find(bid);
    if (it == contract_by_id.end())
      continue;
    const nlohmann::json &b = *it->second;
    if (!b.contains("nodes"))
      continue;
    for (const auto &node : b["nodes"])
    {
      if (
        node.value("nodeType", "") == "FunctionDefinition" &&
        node.value("name", "") == name)
        return &node;
    }
  }
  return nullptr;
}

/// Map a Solidity typeString (from typeDescriptions) to a Solidity source
/// type name.  Handles common types; falls back to the raw typeString for
/// anything exotic.
static std::string soltype_to_source(const std::string &ts)
{
  // "uint256" → "uint256", "address" → "address", etc.
  // "contract Foo" → "Foo"
  if (ts.substr(0, 9) == "contract ")
    return ts.substr(9);
  // "enum Foo.Bar" → "Foo.Bar"
  if (ts.substr(0, 5) == "enum ")
    return ts.substr(5);
  // "struct Foo.Bar" → "Foo.Bar" (may need " memory"/" calldata" at emit
  // site; the caller is responsible for appending the data-location
  // qualifier for reference types).
  if (ts.substr(0, 7) == "struct ")
    return ts.substr(7);
  // mapping(...) — not representable as a function param; skip
  if (ts.substr(0, 7) == "mapping")
    return {};
  // "bytes32", "bool", "string", "int256", etc. — pass through
  return ts;
}

/// Is this Solidity typeString representable as a scalar function return
/// (no `memory`/`storage` location modifier needed)?  Used to decide
/// whether we can synthesise a shadow public getter for a non-public
/// state variable.  Rejects mappings, arrays, structs, strings, dynamic
/// bytes, function types — the shadow-getter strategy only covers
/// primitive, contract, address, enum, and fixed-width bytes types.
static bool is_scalar_returnable_type(const std::string &ts)
{
  if (ts.empty())
    return false;
  if (ts.substr(0, 8) == "mapping(")
    return false;
  if (ts.find('[') != std::string::npos)
    return false;
  if (ts.substr(0, 7) == "struct ")
    return false;
  if (ts.substr(0, 9) == "function ")
    return false;
  if (ts == "string" || ts == "bytes")
    return false;
  return true;
}

/// Solidity 0.5+ requires a data-location qualifier (`memory`/`calldata`)
/// on reference-type parameters: string, bytes, arrays, structs.  Detect
/// those typeStrings so we can add ` memory` on emission.
static bool param_needs_memory_location(const std::string &ts)
{
  if (ts == "string" || ts == "bytes")
    return true;
  if (ts.find('[') != std::string::npos)
    return true;
  if (ts.substr(0, 7) == "struct ")
    return true;
  return false;
}

/// Prefix parameter names to avoid collisions between funcA and funcB.
/// Returns e.g. "uint256 a_a, address a_b".
static std::string emit_prefixed_params(
  const nlohmann::json &func_def,
  const std::string &prefix)
{
  std::string out;
  if (
    !func_def.contains("parameters") ||
    !func_def["parameters"].contains("parameters"))
    return out;
  bool first = true;
  for (const auto &p : func_def["parameters"]["parameters"])
  {
    std::string ts =
      p.value("typeDescriptions", nlohmann::json{}).value("typeString", "");
    std::string name = p.value("name", "");
    std::string st = soltype_to_source(ts);
    if (st.empty())
      continue;
    if (!first)
      out += ", ";
    first = false;
    out += st;
    if (param_needs_memory_location(ts))
      out += " memory";
    out += " ";
    out += prefix + "_" + name;
  }
  return out;
}

/// Emit prefixed argument names: "a_x, a_y".
static std::string emit_prefixed_args(
  const nlohmann::json &func_def,
  const std::string &prefix)
{
  std::string out;
  if (
    !func_def.contains("parameters") ||
    !func_def["parameters"].contains("parameters"))
    return out;
  bool first = true;
  for (const auto &p : func_def["parameters"]["parameters"])
  {
    std::string name = p.value("name", "");
    if (name.empty())
      continue;
    if (!first)
      out += ", ";
    first = false;
    out += prefix + "_" + name;
  }
  return out;
}

/// Collect state variable names and types for assertion generation.
struct StateVar
{
  std::string name;
  std::string type_string;
  std::string visibility;   // "public" / "internal" / "private"
  bool is_mapping;
  bool is_returnable;       // can be read via a Solidity getter (auto or shadow)
  int id;                   // AST id, used to detect references from function bodies
  std::vector<std::string> mapping_key_types; // for (nested) mappings
  std::string value_type_string; // leaf value type for mapping; equals type_string for scalar
};

/// Recursively walk an AST node and collect every `referencedDeclaration`
/// integer.  This captures any identifier whose binding points to another
/// declaration (state vars, locals, functions, modifiers, ...).
static void collect_referenced_decls(
  const nlohmann::json &node,
  std::set<int> &ids)
{
  if (node.is_object())
  {
    auto it = node.find("referencedDeclaration");
    if (it != node.end() && it->is_number_integer())
      ids.insert(it->get<int>());
    for (auto kv = node.begin(); kv != node.end(); ++kv)
      collect_referenced_decls(kv.value(), ids);
  }
  else if (node.is_array())
  {
    for (const auto &child : node)
      collect_referenced_decls(child, ids);
  }
}

/// Walk a Mapping typeName node to collect all key types.
static void collect_mapping_keys(
  const nlohmann::json &type_name,
  std::vector<std::string> &keys)
{
  if (type_name.value("nodeType", "") != "Mapping")
    return;
  if (type_name.contains("keyType"))
    keys.push_back(
      type_name["keyType"]
        .value("typeDescriptions", nlohmann::json{})
        .value("typeString", ""));
  if (type_name.contains("valueType"))
    collect_mapping_keys(type_name["valueType"], keys);
}

/// Walk a Mapping typeName node to its leaf value type string (strip
/// nested Mapping wrappers).  For a non-mapping typeName, returns the
/// node's own typeString.
static std::string extract_leaf_value_type(const nlohmann::json &type_name)
{
  if (type_name.value("nodeType", "") == "Mapping" &&
      type_name.contains("valueType"))
    return extract_leaf_value_type(type_name["valueType"]);
  return type_name.value("typeDescriptions", nlohmann::json{})
    .value("typeString", "");
}

/// Collect state variables declared on `contract_def`.  If `ast` is
/// non-null and the contract has a `linearizedBaseContracts` chain,
/// inherited non-private state vars are folded in as well (in MRO
/// order: target first, then each base).  Private vars on base
/// contracts are dropped — they are inaccessible from the derived
/// contract, so any shadow getter referencing them would fail to
/// compile.  Constants and immutables are skipped everywhere — they
/// cannot TOD.
static std::vector<StateVar> collect_state_vars(
  const nlohmann::json &contract_def,
  const nlohmann::json *ast = nullptr)
{
  auto one_contract = [&](const nlohmann::json &c, bool is_base,
                          std::vector<StateVar> &out) {
    if (!c.contains("nodes"))
      return;
    for (const auto &node : c["nodes"])
    {
      if (node.value("nodeType", "") != "VariableDeclaration")
        continue;
      if (!node.value("stateVariable", false))
        continue;
      if (node.value("constant", false))
        continue;
      const std::string mut = node.value("mutability", "");
      if (mut == "constant" || mut == "immutable")
        continue;
      const std::string vis = node.value("visibility", "internal");
      if (is_base && vis == "private")
        continue;

      StateVar sv;
      sv.name = node.value("name", "");
      sv.type_string =
        node.value("typeDescriptions", nlohmann::json{})
          .value("typeString", "");
      sv.visibility = vis;
      sv.is_mapping = false;
      sv.id = node.value("id", -1);

      if (
        node.contains("typeName") &&
        node["typeName"].value("nodeType", "") == "Mapping")
      {
        sv.is_mapping = true;
        collect_mapping_keys(node["typeName"], sv.mapping_key_types);
        sv.value_type_string = extract_leaf_value_type(node["typeName"]);
        sv.is_returnable = is_scalar_returnable_type(sv.value_type_string);
        for (const auto &k : sv.mapping_key_types)
          if (!is_scalar_returnable_type(k))
          {
            sv.is_returnable = false;
            break;
          }
      }
      else
      {
        sv.value_type_string = sv.type_string;
        sv.is_returnable = is_scalar_returnable_type(sv.type_string);
      }

      if (!sv.name.empty())
        out.push_back(sv);
    }
  };

  std::vector<StateVar> vars;
  one_contract(contract_def, /*is_base=*/false, vars);

  if (!ast || !ast->contains("nodes") || !ast->at("nodes").is_array())
    return vars;
  if (
    !contract_def.contains("linearizedBaseContracts") ||
    !contract_def.at("linearizedBaseContracts").is_array())
    return vars;

  std::map<int, const nlohmann::json *> contract_by_id;
  for (const auto &n : (*ast)["nodes"])
  {
    if (n.value("nodeType", "") != "ContractDefinition")
      continue;
    if (!n.contains("id") || !n["id"].is_number_integer())
      continue;
    contract_by_id[n["id"].get<int>()] = &n;
  }

  const auto &lin = contract_def["linearizedBaseContracts"];
  for (size_t i = 1; i < lin.size(); ++i)
  {
    if (!lin[i].is_number_integer())
      continue;
    int bid = lin[i].get<int>();
    auto it = contract_by_id.find(bid);
    if (it == contract_by_id.end())
      continue;
    one_contract(*it->second, /*is_base=*/true, vars);
  }
  return vars;
}

/// Collect parameters by type from a function (prefixed names).
static std::vector<std::string> collect_params_by_type(
  const nlohmann::json &func_def,
  const std::string &prefix,
  const std::string &type_string)
{
  std::vector<std::string> keys;
  if (
    !func_def.contains("parameters") ||
    !func_def["parameters"].contains("parameters"))
    return keys;
  for (const auto &p : func_def["parameters"]["parameters"])
  {
    std::string ts =
      p.value("typeDescriptions", nlohmann::json{}).value("typeString", "");
    std::string name = p.value("name", "");
    if (ts == type_string && !name.empty())
      keys.push_back(prefix + "_" + name);
  }
  return keys;
}

/// Rename all occurrences of `old_name` as a whole word in `text`.
static std::string rename_contract(
  const std::string &text,
  const std::string &old_name,
  const std::string &new_name)
{
  // Use word-boundary regex to avoid false matches in substrings.
  std::regex re("\\b" + old_name + "\\b");
  return std::regex_replace(text, re, new_name);
}

/// Inject a synthetic `function __tod_bal() public view returns (uint) {
/// return address(this).balance; }` immediately before the contract's
/// closing `}`.
///
/// Needed because the TOD harness uses `new`-allocated dynamic instances
/// (c1 / c2), but ESBMC's Two-Copy Rename routes method calls through
/// the renamed contract's *singleton* (_ESBMC_Object_V_C1) — state
/// reads on the dynamic pointer (`c1->$balance`) therefore see a
/// never-updated nondet value, while the singleton tracks the real
/// balance.  By asserting equality through a call (`c1.__tod_bal()`)
/// we go through the dispatch; inside the callee `address(this).balance`
/// resolves against `this = &singleton` (see the AddressMemberCall fix
/// in solidity_convert_expr.cpp) and returns the up-to-date value.
static std::string inject_tod_bal_getter(const std::string &contract_src)
{
  auto last_brace = contract_src.rfind('}');
  if (last_brace == std::string::npos)
    return contract_src;
  // Inject TWO synthetic getters:
  //   __tod_bal()  -> singleton's $balance (via address(this).balance)
  //   __tod_addr() -> singleton's $address (via address(this))
  // Both go through the contract-call dispatch so the result reflects
  // the singleton state, not the dynamic-struct field accessed
  // directly via `c1->$balance` / `address(c1)` from outside.  The
  // address getter is needed for the harness's isolation guards:
  // `address(c1)` in the harness reads c1's dynamic $address (one
  // nondet), while the transfer dispatcher matches on the singleton's
  // $address (a different nondet).  Routing the require() comparison
  // through __tod_addr() makes both sides see the SAME (singleton)
  // address.
  std::string injected =
    "\n    function __tod_bal() public view returns (uint) { "
    "return address(this).balance; }\n"
    "    function __tod_addr() public view returns (address) { "
    "return address(this); }\n";
  return contract_src.substr(0, last_brace) + injected +
         contract_src.substr(last_brace);
}

/// Inject shadow public getters for every non-public state variable whose
/// type is scalar-returnable.  A state var `uint256 x` declared `internal`
/// or `private` cannot be read through `c.x()` from the harness — Solidity
/// only auto-generates getters for `public` state vars.  We side-step this
/// by emitting
///   function __tod_get_x() public view returns (uint256) { return x; }
/// (scalar case) or
///   function __tod_get_m(K1 a, K2 b) public view returns (V) { return m[a][b]; }
/// (mapping case) into the contract source, which compiles cleanly under
/// a standard Solidity compiler and exposes the underlying storage slot
/// through a callable interface.  The harness then routes reads through
/// `c.__tod_get_<name>(...)` instead of the auto-generated `c.<name>(...)`.
///
/// Names are prefixed with `__tod_get_` to avoid collisions with any
/// user-authored identifiers.  Non-returnable state vars (struct, array,
/// string, dynamic bytes, mapping with non-scalar leaf, function types)
/// are left as-is — the emitter already has a `// Skipped` branch for
/// each unsupported shape.
///
/// Returnable-but-public state vars are skipped here because the Solidity
/// compiler already synthesises the equivalent getter automatically.
static std::string inject_shadow_getters(
  const std::string &contract_src,
  const std::vector<StateVar> &state_vars)
{
  std::string injected;
  for (const auto &sv : state_vars)
  {
    if (sv.visibility == "public")
      continue;
    if (!sv.is_returnable)
      continue;

    const std::string ret_type = soltype_to_source(sv.value_type_string);
    if (ret_type.empty())
      continue;

    if (!sv.is_mapping)
    {
      injected += "    function __tod_get_" + sv.name +
                  "() public view returns (" + ret_type + ") { return " +
                  sv.name + "; }\n";
    }
    else
    {
      std::string params;
      std::string keys;
      for (size_t i = 0; i < sv.mapping_key_types.size(); ++i)
      {
        const std::string kt = soltype_to_source(sv.mapping_key_types[i]);
        if (kt.empty())
        {
          params.clear();
          break;
        }
        if (i)
        {
          params += ", ";
          keys += "][";
        }
        params += kt + " __k" + std::to_string(i);
        keys += "__k" + std::to_string(i);
      }
      if (params.empty())
        continue;
      injected += "    function __tod_get_" + sv.name + "(" + params +
                  ") public view returns (" + ret_type + ") { return " +
                  sv.name + "[" + keys + "]; }\n";
    }
  }
  if (injected.empty())
    return contract_src;
  auto last_brace = contract_src.rfind('}');
  if (last_brace == std::string::npos)
    return contract_src;
  return contract_src.substr(0, last_brace) + "\n" + injected +
         contract_src.substr(last_brace);
}

/// Return the getter name the harness should use for a state var: the
/// Solidity-synthesised name for public vars, or the shadow name we
/// injected for internal/private vars.
static std::string state_var_getter_name(const StateVar &sv)
{
  return sv.visibility == "public" ? sv.name : ("__tod_get_" + sv.name);
}

/// Extract constructor parameters from a ContractDefinition.
static const nlohmann::json *find_constructor(const nlohmann::json &contract)
{
  if (!contract.contains("nodes"))
    return nullptr;
  for (const auto &node : contract["nodes"])
  {
    if (
      node.value("nodeType", "") == "FunctionDefinition" &&
      node.value("kind", "") == "constructor")
      return &node;
  }
  return nullptr;
}

/// Walk `node` recursively and return true if any subtree contains a
/// revert-inducing construct: an explicit `require(...)` / `revert(...)` /
/// `assert(...)` call, a `revert SomeError(...)` statement, or an explicit
/// throw.  Used to decide whether to wrap a call-site in `try/catch` in the
/// TOD harness — non-reverting callees get a raw call to avoid the nondet
/// "catch arm" over-approximating real EVM behavior.
///
/// Limitation: this is a conservative SYNTACTIC check.  It does NOT detect
/// reverts arising from overflow, failed external calls, or callees of
/// callees.  For the callee with overflow-checked arithmetic in 0.8+, a raw
/// call is still "safe enough" here because the overflow path also emits
/// `assume(false)` which prunes just that path — the harness test()
/// continues along the non-overflow path as long as at least one ordering
/// keeps running.  Reverts inside *callees-of-callees* are the known soft
/// spot and are still handled by the try/catch path whenever the top-level
/// function has any syntactic revert construct.
static bool body_may_revert_explicitly(const nlohmann::json &node)
{
  if (node.is_object())
  {
    const std::string nt = node.value("nodeType", "");
    if (nt == "FunctionCall" && node.contains("expression"))
    {
      const auto &expr = node["expression"];
      if (expr.is_object() && expr.value("nodeType", "") == "Identifier")
      {
        const std::string name = expr.value("name", "");
        if (name == "require" || name == "revert" || name == "assert")
          return true;
      }
    }
    if (nt == "RevertStatement" || nt == "Throw")
      return true;
    for (auto it = node.begin(); it != node.end(); ++it)
    {
      if (body_may_revert_explicitly(it.value()))
        return true;
    }
  }
  else if (node.is_array())
  {
    for (const auto &el : node)
      if (body_may_revert_explicitly(el))
        return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Per-pair harness contract emission
// ---------------------------------------------------------------------------

/// Balance mode harness: uses two renamed copies of the target contract
/// (<contract>_C1 / <contract>_C2) so ETH transfers on distinct singletons
/// do not alias.  Kept as the balance-TOD backend because the ETH balance
/// model lives on `_ESBMC_Object_<C>.$balance` and the harness needs two
/// separate singletons.
static bool emit_harness_contract(
  std::ostringstream &out,
  const nlohmann::json &cdef,
  const nlohmann::json &ast,
  const std::vector<StateVar> &all_state_vars,
  const std::map<std::string, solidity_tod::RWSet> &rw_by_name,
  const nlohmann::json *ctor,
  const std::string &c1_name,
  const std::string &c2_name,
  const std::string &func_a,
  const std::string &func_b)
{
  const nlohmann::json *fa = find_function(cdef, func_a, &ast);
  const nlohmann::json *fb = find_function(cdef, func_b, &ast);
  if (!fa || !fb)
  {
    log_error(
      "TOD harness: function '{}' or '{}' not found",
      func_a,
      func_b);
    return false;
  }

  // Phase 3: keep only state vars referenced by BOTH functions, using the
  // call-graph-closed footprint so writes hidden in internal helpers count.
  std::set<int> footprint_a, footprint_b;
  auto it_a = rw_by_name.find(func_a);
  if (it_a != rw_by_name.end())
  {
    footprint_a.insert(it_a->second.reads.begin(), it_a->second.reads.end());
    footprint_a.insert(it_a->second.writes.begin(), it_a->second.writes.end());
  }
  auto it_b = rw_by_name.find(func_b);
  if (it_b != rw_by_name.end())
  {
    footprint_b.insert(it_b->second.reads.begin(), it_b->second.reads.end());
    footprint_b.insert(it_b->second.writes.begin(), it_b->second.writes.end());
  }
  std::set<int> shared_ids;
  for (int id : footprint_a)
    if (footprint_b.count(id))
      shared_ids.insert(id);

  std::vector<StateVar> shared_vars;
  std::vector<std::string> skipped_vars;
  for (const auto &sv : all_state_vars)
  {
    if (sv.id >= 0 && shared_ids.count(sv.id))
      shared_vars.push_back(sv);
    else
      skipped_vars.push_back(sv.name);
  }
  // Phase 4: TOD-Balance.  If the closed footprint puts the virtual
  // __balance token in the shared set, both functions touch ETH balance —
  // emit `assert(address(c1).balance == address(c2).balance)`.
  bool emit_balance_assert =
    shared_ids.count(solidity_tod::kBalanceId) != 0;

  // Constructor params
  std::string ctor_params_decl;
  std::string ctor_args;
  if (ctor)
  {
    ctor_params_decl = emit_prefixed_params(*ctor, "ctor");
    ctor_args = emit_prefixed_args(*ctor, "ctor");
  }

  // Function params (prefixed to avoid name collisions).
  std::string fa_params = emit_prefixed_params(*fa, "a");
  std::string fb_params = emit_prefixed_params(*fb, "b");
  std::string fa_args = emit_prefixed_args(*fa, "a");
  std::string fb_args = emit_prefixed_args(*fb, "b");

  // For each mapping key type, gather all matching params from both funcs.
  std::map<std::string, std::vector<std::string>> params_by_type;
  for (const auto &sv : shared_vars)
  {
    if (!sv.is_mapping)
      continue;
    for (const auto &kt : sv.mapping_key_types)
    {
      if (params_by_type.count(kt))
        continue;
      auto ka = collect_params_by_type(*fa, "a", kt);
      auto kb = collect_params_by_type(*fb, "b", kt);
      std::vector<std::string> all;
      all.insert(all.end(), ka.begin(), ka.end());
      all.insert(all.end(), kb.begin(), kb.end());
      params_by_type[kt] = all;
    }
  }

  // Per-pair audit comment block.
  out << "// ----- " << func_a << " vs " << func_b << " -----\n";
  out << "// Targeted state variables (referenced by BOTH functions):";
  if (shared_vars.empty())
    out << " <none>\n";
  else
  {
    out << "\n";
    for (const auto &sv : shared_vars)
      out << "//   - " << sv.name << "\n";
  }
  if (!skipped_vars.empty())
  {
    out << "// Skipped (touched by at most one function):\n";
    for (const auto &n : skipped_vars)
      out << "//   - " << n << "\n";
  }
  if (emit_balance_assert)
    out << "// Plus: address(this).balance (TOD-Balance check)\n";

  std::string harness_name = "TOD_" + func_a + "_" + func_b;
  // For TOD-Balance with a payable constructor, fund both copies with the
  // same nondet initial balance so transfers in the function bodies don't
  // get pruned by the `balance < value -> assume(false)` revert model.
  // Requires the test() function itself to be payable so msg.value can
  // cover both `new C{value: __initBal}()` calls.
  bool ctor_payable =
    ctor && ctor->value("stateMutability", "") == "payable";
  bool fund_copies = emit_balance_assert && ctor_payable;

  out << "contract " << harness_name << " {\n";
  out << "    function test(\n";

  std::vector<std::string> all_params;
  if (fund_copies)
    all_params.push_back("uint __initBal");
  if (!ctor_params_decl.empty())
    all_params.push_back(ctor_params_decl);
  if (!fa_params.empty())
    all_params.push_back(fa_params);
  if (!fb_params.empty())
    all_params.push_back(fb_params);

  for (size_t i = 0; i < all_params.size(); ++i)
  {
    out << "        " << all_params[i];
    if (i + 1 < all_params.size())
      out << ",";
    out << "\n";
  }
  out << "    ) public" << (fund_copies ? " payable {\n" : " {\n");

  if (fund_copies)
  {
    out << "        " << c1_name << " c1 = new " << c1_name
        << "{value: __initBal}(" << ctor_args << ");\n";
    out << "        " << c2_name << " c2 = new " << c2_name
        << "{value: __initBal}(" << ctor_args << ");\n\n";
  }
  else
  {
    out << "        " << c1_name << " c1 = new " << c1_name << "("
        << ctor_args << ");\n";
    out << "        " << c2_name << " c2 = new " << c2_name << "("
        << ctor_args << ");\n\n";
  }

  // For TOD-Balance, exclude harness-internal addresses from any
  // address-typed parameter so the two copies stay isolated under
  // ETH transfers.  Without this guard, e.g. a_to == address(c1)
  // would route c2's payment back into c1, breaking the symmetry the
  // harness relies on.
  if (emit_balance_assert)
  {
    auto collect_addr_params = [](const nlohmann::json &fdef,
                                  const std::string &prefix,
                                  std::vector<std::string> &out_list) {
      if (!fdef.contains("parameters") ||
          !fdef["parameters"].contains("parameters"))
        return;
      for (const auto &p : fdef["parameters"]["parameters"])
      {
        const std::string ts = p.value("typeDescriptions", nlohmann::json{})
                                 .value("typeString", "");
        const std::string name = p.value("name", "");
        if (
          (ts == "address" || ts == "address payable") && !name.empty())
          out_list.push_back(prefix + "_" + name);
      }
    };
    std::vector<std::string> addr_params;
    collect_addr_params(*fa, "a", addr_params);
    collect_addr_params(*fb, "b", addr_params);
    if (ctor)
      collect_addr_params(*ctor, "ctor", addr_params);
    if (!addr_params.empty())
    {
      // Cache the singleton-side addresses through __tod_addr() once;
      // each subsequent require() against the cached locals is just an
      // address compare.
      out << "        address __c1_addr = c1.__tod_addr();\n";
      out << "        address __c2_addr = c2.__tod_addr();\n";
      for (const auto &p : addr_params)
      {
        out << "        require(" << p << " != __c1_addr && " << p
            << " != __c2_addr, \"isolate copies\");\n";
      }
      out << "\n";
    }
  }

  // Call-site wrapping: for a callee with an explicit `require`/`revert`/
  // `assert`, wrap in `try/catch` so a revert in the first call does NOT
  // abort `test()` and swallow the equality assertions (which would yield
  // a vacuous VERIFICATION SUCCESSFUL on require-guarded TODs).  For a
  // callee with NO syntactic revert, keep the raw call — the catch arm is
  // modelled nondet by ESBMC's `TryStatement` converter and would
  // over-approximate into false TOD reports (the `tod_balance_pass` case,
  // where both calls always succeed but a phantom catch path creates a
  // state mismatch).  Trade-off: catches reverts in explicit-guard bodies
  // only; reverts arising from overflow or nested callees still ride on
  // the raw call's `assume(false)` path pruning, which is unchanged.
  const bool wrap_a = body_may_revert_explicitly(*fa);
  const bool wrap_b = body_may_revert_explicitly(*fb);
  auto emit_call = [&](std::ostringstream &o, const std::string &c,
                       const std::string &fn, const std::string &args,
                       bool wrap) {
    if (wrap)
      o << "        try " << c << "." << fn << "(" << args
        << ") {} catch {}\n";
    else
      o << "        " << c << "." << fn << "(" << args << ");\n";
  };

  out << "        // Order 1: " << func_a << " then " << func_b << "\n";
  emit_call(out, "c1", func_a, fa_args, wrap_a);
  emit_call(out, "c1", func_b, fb_args, wrap_b);
  out << "\n";

  out << "        // Order 2: " << func_b << " then " << func_a << "\n";
  emit_call(out, "c2", func_b, fb_args, wrap_b);
  emit_call(out, "c2", func_a, fa_args, wrap_a);
  out << "\n";

  out << "        // State comparison — if any assert fails, TOD exists\n";
  for (const auto &sv : shared_vars)
  {
    if (!sv.is_returnable)
    {
      out << "        // Skipped: " << sv.name
          << " (type not reachable via a Solidity getter)\n";
      continue;
    }
    if (sv.is_mapping && !sv.mapping_key_types.empty())
    {
      std::vector<std::vector<std::string>> key_sets;
      bool skip = false;
      for (const auto &kt : sv.mapping_key_types)
      {
        if (!params_by_type.count(kt) || params_by_type[kt].empty())
        {
          skip = true;
          break;
        }
        key_sets.push_back(params_by_type[kt]);
      }
      if (skip)
      {
        out << "        // Skipped: " << sv.name
            << " (no matching key params)\n";
        continue;
      }

      const std::string getter = state_var_getter_name(sv);
      if (key_sets.size() == 1)
      {
        for (const auto &k : key_sets[0])
          out << "        assert(c1." << getter << "(" << k << ")"
              << " == c2." << getter << "(" << k << "));\n";
      }
      else if (key_sets.size() == 2)
      {
        for (const auto &k1 : key_sets[0])
          for (const auto &k2 : key_sets[1])
            out << "        assert(c1." << getter << "(" << k1 << ", "
                << k2 << ") == c2." << getter << "(" << k1 << ", "
                << k2 << "));\n";
      }
      else
      {
        out << "        // Skipped: " << sv.name
            << " (deeply nested mapping, " << key_sets.size()
            << " levels)\n";
      }
    }
    else if (!sv.is_mapping)
    {
      const std::string getter = state_var_getter_name(sv);
      out << "        assert(c1." << getter << "()"
          << " == c2." << getter << "());\n";
    }
  }
  if (emit_balance_assert)
    out << "        assert(c1.__tod_bal() == c2.__tod_bal());\n";

  out << "    }\n";
  out << "}\n\n";

  if (shared_vars.empty() && !emit_balance_assert)
    log_warning(
      "TOD harness: no state variable nor ETH balance is touched "
      "by both '{}' and '{}'.  The harness contains no equality "
      "assertion; verification will trivially succeed.",
      func_a,
      func_b);
  return true;
}

/// Race mode harness: single target contract + `function test(C c1, C c2,
/// ...)` parameters.  Independent storage per `c1`/`c2` comes from
/// ESBMC's --contract-param-fresh flag (assign_param_nondet allocates a
/// fresh heap instance per contract-typed parameter).  The `ctor_args`
/// parameter list is still threaded through so the harness body can call
/// `c1.someCtorPublicGetter()` with the same constants if that ever
/// becomes useful — for the basic race check we only exercise f_a and
/// f_b, not the ctor.
///
/// TOD classification is surfaced through helper functions named
/// `__tod_race_check`/`__tod_balance_check`; their bodies are a plain
/// `assert(cond)` so Solidity compiles, and the function name appears
/// on the ESBMC counter-example stack, distinguishing race from balance
/// TOD in the verdict output.
static bool emit_harness_contract_race(
  std::ostringstream &out,
  const nlohmann::json &cdef,
  const nlohmann::json &ast,
  const std::vector<StateVar> &all_state_vars,
  const std::map<std::string, solidity_tod::RWSet> &rw_by_name,
  const nlohmann::json *ctor,
  const std::string &contract,
  const std::string &func_a,
  const std::string &func_b)
{
  const nlohmann::json *fa = find_function(cdef, func_a, &ast);
  const nlohmann::json *fb = find_function(cdef, func_b, &ast);
  if (!fa || !fb)
  {
    log_error(
      "TOD harness: function '{}' or '{}' not found", func_a, func_b);
    return false;
  }

  // Assert on the UNION of writes: a TOD-Race exposes divergence on any
  // state variable WRITTEN by either function, not just the intersection.
  // Pair SELECTION still uses the overlap criterion — but the witness
  // check compares the full post-state footprint so we don't miss cases
  // like `f_a writes pot, f_b reads pot and writes winner` where the
  // divergence is on winner (touched by only one side).
  std::set<int> asserted_ids;
  auto it_a = rw_by_name.find(func_a);
  if (it_a != rw_by_name.end())
    asserted_ids.insert(
      it_a->second.writes.begin(), it_a->second.writes.end());
  auto it_b = rw_by_name.find(func_b);
  if (it_b != rw_by_name.end())
    asserted_ids.insert(
      it_b->second.writes.begin(), it_b->second.writes.end());

  std::vector<StateVar> shared_vars;
  for (const auto &sv : all_state_vars)
    if (sv.id >= 0 && asserted_ids.count(sv.id))
      shared_vars.push_back(sv);

  // Function params (prefixed so fa/fb argument names don't collide).
  std::string fa_params = emit_prefixed_params(*fa, "a");
  std::string fb_params = emit_prefixed_params(*fb, "b");
  std::string fa_args = emit_prefixed_args(*fa, "a");
  std::string fb_args = emit_prefixed_args(*fb, "b");

  // Per mapping key type, gather matching params from both functions.
  std::map<std::string, std::vector<std::string>> params_by_type;
  for (const auto &sv : shared_vars)
  {
    if (!sv.is_mapping)
      continue;
    for (const auto &kt : sv.mapping_key_types)
    {
      if (params_by_type.count(kt))
        continue;
      auto ka = collect_params_by_type(*fa, "a", kt);
      auto kb = collect_params_by_type(*fb, "b", kt);
      std::vector<std::string> all;
      all.insert(all.end(), ka.begin(), ka.end());
      all.insert(all.end(), kb.begin(), kb.end());
      // For address keys, also enumerate `address(this)`: functions called
      // through c1/c2 see msg.sender == address(this) of the harness, so
      // mappings keyed by msg.sender (e.g. ERC20 allowed[msg.sender][...])
      // land in slot (address(this), _spender).  Without this the writes
      // from approve / increaseApproval are never asserted on.
      if (kt == "address" || kt == "address payable")
        all.push_back("address(this)");
      params_by_type[kt] = all;
    }
  }

  out << "// ----- " << func_a << " vs " << func_b << " -----\n";
  out << "// Shared state variables (touched by both):\n";
  if (shared_vars.empty())
    out << "//   <none — assertion is trivially true>\n";
  for (const auto &sv : shared_vars)
    out << "//   - " << sv.name << "\n";

  // Constructor params (threaded as harness test-function params) so
  // `new C(ctor_a, ctor_b, ...)` compiles for contracts with a
  // non-nullary ctor.  For contracts without a ctor, this is empty.
  std::string ctor_params_decl;
  std::string ctor_args;
  if (ctor)
  {
    ctor_params_decl = emit_prefixed_params(*ctor, "ctor");
    ctor_args = emit_prefixed_args(*ctor, "ctor");
  }

  std::string harness_name = "TOD_" + func_a + "_" + func_b;
  out << "contract " << harness_name << " {\n";
  out << "    function test(";
  // Only scalar / mapping-key params now — c1/c2 are materialised inside
  // the body via the __ESOL_* intrinsics, so the harness dump is fully
  // self-describing and requires no hidden frontend injection.
  std::vector<std::string> all_params;
  if (!ctor_params_decl.empty())
    all_params.push_back(ctor_params_decl);
  if (!fa_params.empty())
    all_params.push_back(fa_params);
  if (!fb_params.empty())
    all_params.push_back(fb_params);
  if (all_params.empty())
    out << ") public {\n";
  else
  {
    out << "\n";
    for (size_t i = 0; i < all_params.size(); ++i)
    {
      out << "        " << all_params[i];
      if (i + 1 < all_params.size())
        out << ",";
      out << "\n";
    }
    out << "    ) public {\n";
  }
  // Canonical three-step setup: allocate, nondet-drive to any reachable
  // pre-race state S, then deep-copy so c1 and c2 observe the same S.
  // The intrinsics are the stable ESBMC ABI (see get_call_expr in
  // solidity_convert_expr.cpp).
  out << "        " << contract << " c1 = new " << contract << "("
      << ctor_args << ");\n";
  out << "        __ESOL_nondet_state_forward(c1);\n";
  out << "        " << contract << " c2 = __ESOL_deep_copy(c1);\n";
  // Require distinct $address so any mapping state var (keyed by
  // (addr, key) in the _ESBMC_Mapping store) does not alias between
  // the two deployments.  __ESOL_deep_copy already mints a fresh
  // address and asserts disjointness, so this require() is a
  // self-documenting belt-and-braces check rather than a real guard.
  out << "        require(address(c1) != address(c2), \"isolate c1/c2\");\n";

  const bool wrap_a = body_may_revert_explicitly(*fa);
  const bool wrap_b = body_may_revert_explicitly(*fb);
  auto emit_call = [&](std::ostringstream &o, const std::string &c,
                       const std::string &fn, const std::string &args,
                       bool wrap) {
    if (wrap)
      o << "        try " << c << "." << fn << "(" << args
        << ") {} catch {}\n";
    else
      o << "        " << c << "." << fn << "(" << args << ");\n";
  };

  out << "        // Order 1: c1 runs " << func_a << " then " << func_b
      << "\n";
  emit_call(out, "c1", func_a, fa_args, wrap_a);
  emit_call(out, "c1", func_b, fb_args, wrap_b);
  out << "\n        // Order 2: c2 runs " << func_b << " then " << func_a
      << "\n";
  emit_call(out, "c2", func_b, fb_args, wrap_b);
  emit_call(out, "c2", func_a, fa_args, wrap_a);
  out << "\n        // Race check: if any shared state differs the pair is "
         "order-dependent\n";

  for (const auto &sv : shared_vars)
  {
    if (!sv.is_returnable)
    {
      out << "        // Skipped: " << sv.name
          << " (type not reachable via a Solidity getter)\n";
      continue;
    }
    if (sv.is_mapping && !sv.mapping_key_types.empty())
    {
      std::vector<std::vector<std::string>> key_sets;
      bool skip = false;
      for (const auto &kt : sv.mapping_key_types)
      {
        if (!params_by_type.count(kt) || params_by_type[kt].empty())
        {
          skip = true;
          break;
        }
        key_sets.push_back(params_by_type[kt]);
      }
      if (skip)
      {
        out << "        // Skipped: " << sv.name
            << " (no matching key params)\n";
        continue;
      }
      const std::string getter = state_var_getter_name(sv);
      if (key_sets.size() == 1)
      {
        for (const auto &k : key_sets[0])
          out << "        __tod_race_check(c1." << getter << "(" << k
              << ") == c2." << getter << "(" << k << "));\n";
      }
      else if (key_sets.size() == 2)
      {
        for (const auto &k1 : key_sets[0])
          for (const auto &k2 : key_sets[1])
            out << "        __tod_race_check(c1." << getter << "(" << k1
                << ", " << k2 << ") == c2." << getter << "(" << k1 << ", "
                << k2 << "));\n";
      }
      else
      {
        out << "        // Skipped: " << sv.name
            << " (deeply nested mapping, " << key_sets.size()
            << " levels)\n";
      }
    }
    else if (!sv.is_mapping)
    {
      const std::string getter = state_var_getter_name(sv);
      out << "        __tod_race_check(c1." << getter << "() == c2."
          << getter << "());\n";
    }
  }

  out << "    }\n";
  out << "}\n\n";

  if (shared_vars.empty())
    log_warning(
      "TOD race harness: no state variable is touched by both '{}' "
      "and '{}'.  Harness contains no equality assertion; verification "
      "trivially succeeds.",
      func_a,
      func_b);
  return true;
}

// ---------------------------------------------------------------------------
// Top-level generator (single pair -> single .sol file)
// ---------------------------------------------------------------------------

std::string generate_tod_harness(
  const std::string &sol_source,
  const nlohmann::json &ast,
  const std::string &contract,
  const std::string &func_a,
  const std::string &func_b,
  TodHarnessMode mode)
{
  const nlohmann::json *cdef = find_contract(ast, contract);
  if (!cdef)
  {
    log_error("TOD harness: contract '{}' not found in AST", contract);
    return {};
  }

  std::string contract_src = extract_src(sol_source, cdef->value("src", ""));
  if (contract_src.empty())
  {
    log_error("TOD harness: cannot extract contract source text");
    return {};
  }

  auto state_vars = collect_state_vars(*cdef, &ast);
  auto rw_by_name = solidity_tod::compute_rw_sets(*cdef, &ast);

  std::ostringstream out;
  out << "// Auto-generated TOD (Transaction Order Dependence) harness\n";
  out << "// Contract: " << contract << "\n";
  out << "// Pair:     " << func_a << " vs " << func_b << "\n";
  out << "// Mode:     "
      << (mode == TodHarnessMode::Balance ? "balance" : "race") << "\n\n";
  out << "// SPDX-License-Identifier: MIT\n";
  out << "pragma solidity >=0.8.0;\n\n";

  // Classification helpers: the ESBMC counter-example prints the callee
  // name, so a failed assert inside __tod_race_check / __tod_balance_check
  // makes the verdict self-describing without needing a 2-arg intrinsic.
  out << "// TOD classification helpers.  An assertion failure inside one\n"
         "// of these functions tells the user which TOD category fired.\n"
         "function __tod_race_check(bool cond) pure {\n"
         "    assert(cond); // TOD-Race: non-commutative state update\n"
         "}\n"
         "function __tod_balance_check(bool cond) pure {\n"
         "    assert(cond); // TOD-Balance: order-dependent ETH movement\n"
         "}\n\n";

  // ESBMC __ESOL_* intrinsic stubs.  solc requires a concrete body on a
  // free function; ESBMC's Solidity frontend recognises the __ESOL_
  // prefix in get_noncontract_defition and drops the body, then replaces
  // every call site in get_call_expr with the corresponding internal
  // helper (build_esol_state_forward_helper / build_tod_clone_helper).
  // Users copying this harness into their own projects can lift the
  // stubs verbatim.
  //
  // Race mode only: balance mode still uses the dual-copy rename
  // approach (Bal_C1/Bal_C2) for singleton-balance isolation, so the
  // original `contract` identifier isn't present in the emitted file.
  if (mode != TodHarnessMode::Balance)
  {
    out << "// ESBMC intrinsic stubs (the frontend ignores the bodies).\n"
           "function __ESOL_nondet_state_forward(" << contract << " c) {\n"
           "    // replaced by ESBMC with a bounded nondet-dispatch loop\n"
           "    // over c's public/external methods.\n"
           "}\n"
           "function __ESOL_deep_copy(" << contract << " src) pure returns ("
        << contract << ") {\n"
           "    // replaced by ESBMC with _ESBMC_clone_" << contract
        << ": per-field deep copy of *src\n"
           "    // into a fresh instance with a distinct $address and\n"
           "    // independent heap-allocated array buffers.\n"
           "    return src;\n"
           "}\n\n";
  }

  std::string deps = collect_dependency_definitions(ast, sol_source, contract);
  if (!deps.empty())
  {
    out << "// ===== Dependencies =====\n";
    out << deps;
    out << "// ===== End dependencies =====\n\n";
  }

  if (mode == TodHarnessMode::Balance)
  {
    // Balance mode keeps the two-copy rename approach: each copy owns its
    // own _ESBMC_Object_<Cx> singleton so ETH transfers routed through the
    // address dispatcher land in distinct balance slots.
    std::string c1_name = contract + "_C1";
    std::string c2_name = contract + "_C2";
    std::string copy1 = rename_contract(contract_src, contract, c1_name);
    std::string copy2 = rename_contract(contract_src, contract, c2_name);
    copy1 = inject_tod_bal_getter(copy1);
    copy2 = inject_tod_bal_getter(copy2);
    copy1 = inject_shadow_getters(copy1, state_vars);
    copy2 = inject_shadow_getters(copy2, state_vars);
    const nlohmann::json *ctor = find_constructor(*cdef);
    out << "// ===== Copy 1 =====\n" << copy1 << "\n\n";
    out << "// ===== Copy 2 =====\n" << copy2 << "\n\n";
    out << "// ===== TOD harness =====\n";
    if (!emit_harness_contract(
          out,
          *cdef,
          ast,
          state_vars,
          rw_by_name,
          ctor,
          c1_name,
          c2_name,
          func_a,
          func_b))
      return {};
  }
  else
  {
    // Race mode: single contract, param-form harness.  --contract-param-fresh
    // gives c1 and c2 independent storage.
    std::string target_src = inject_shadow_getters(contract_src, state_vars);
    out << "// ===== Target contract =====\n" << target_src << "\n\n";
    out << "// ===== TOD harness =====\n";
    const nlohmann::json *ctor = find_constructor(*cdef);
    if (!emit_harness_contract_race(
          out, *cdef, ast, state_vars, rw_by_name, ctor, contract, func_a,
          func_b))
      return {};
  }

  return out.str();
}

std::string generate_tod_harness_multi(
  const std::string &sol_source,
  const nlohmann::json &ast,
  const std::string &contract,
  const std::vector<std::pair<std::string, std::string>> &pairs,
  TodHarnessMode mode)
{
  // Retained for the single-pair case.  Auto mode is handled by the CLI
  // driver, which invokes generate_tod_harness() per pair and writes a
  // distinct .sol file per pair.
  if (pairs.size() != 1)
  {
    log_error(
      "generate_tod_harness_multi: multi-pair single-file harnesses are no "
      "longer supported; the CLI emits one .sol per pair instead");
    return {};
  }
  return generate_tod_harness(
    sol_source, ast, contract, pairs[0].first, pairs[0].second, mode);
}
