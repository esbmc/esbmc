#include <solidity-frontend/solidity_tod_analysis.h>

#include <algorithm>

namespace solidity_tod
{

namespace
{

/// Snapshot of the contract's structural decls we need for the walk.
struct ContractIndex
{
  // state variable id -> name (only used for diagnostics)
  std::map<int, std::string> state_var_ids;
  // internal function id -> name (intra-contract callgraph nodes)
  std::map<int, std::string> internal_fn_ids;
  // modifier id -> name (treated as inline-callable, included in callgraph)
  std::map<int, std::string> modifier_ids;
  // function id -> body json pointer (FunctionDefinition or ModifierDefinition)
  std::map<int, const nlohmann::json *> body_by_id;
  // function id -> name for any callable inside the contract (used both for
  // walking and for callgraph lookup)
  std::map<int, std::string> all_callable_names;
};

/// Index a single ContractDefinition node.  Used as the inner loop of
/// `index_contract`; the `is_base` flag tells us to drop `private`
/// state vars (they are inaccessible from a derived contract per
/// Solidity's visibility rules) and to skip function/modifier names we
/// already saw in a more-derived contract (MRO override — the leaf's
/// version shadows the base's).
static void index_one(
  ContractIndex &idx,
  const nlohmann::json &contract_def,
  bool is_base,
  std::set<std::string> &seen_fn_names,
  std::set<std::string> &seen_mod_names)
{
  if (!contract_def.contains("nodes"))
    return;
  for (const auto &node : contract_def["nodes"])
  {
    const std::string nt = node.value("nodeType", "");
    int id = node.value("id", -1);
    if (id < 0)
      continue;

    if (nt == "VariableDeclaration" && node.value("stateVariable", false))
    {
      if (is_base && node.value("visibility", "") == "private")
        continue;
      idx.state_var_ids[id] = node.value("name", "");
    }
    else if (nt == "FunctionDefinition")
    {
      const std::string fname = node.value("name", "");
      // Name-based MRO override: the first occurrence walking
      // target → parents is the one that dispatches at runtime.
      // Overloading (same name, different signatures) is treated
      // conservatively here — an overloaded pair in a base will be
      // dropped if the derived contract defines any same-named
      // function.  Acceptable: the derived-only version still gets
      // indexed, and overloading-driven races are rare in practice.
      if (!fname.empty() && !seen_fn_names.insert(fname).second)
        continue;
      idx.internal_fn_ids[id] = fname;
      idx.all_callable_names[id] = fname;
      idx.body_by_id[id] = &node;
    }
    else if (nt == "ModifierDefinition")
    {
      const std::string mname = node.value("name", "");
      if (!mname.empty() && !seen_mod_names.insert(mname).second)
        continue;
      idx.modifier_ids[id] = mname;
      idx.all_callable_names[id] = mname;
      idx.body_by_id[id] = &node;
    }
  }
}

/// Build an index of the target contract.  When `ast` is non-null and
/// `contract_def.linearizedBaseContracts` is populated, every reachable
/// base contract is folded into the index as well (in MRO order: the
/// target first, then its bases left-to-right).  Functions/modifiers
/// are de-duplicated by name so a leaf override does not introduce
/// ghost pairs against the base's shadowed version.
static ContractIndex index_contract(
  const nlohmann::json &contract_def,
  const nlohmann::json *ast = nullptr)
{
  ContractIndex idx;
  std::set<std::string> seen_fn_names;
  std::set<std::string> seen_mod_names;

  // Always index the target first — it wins every MRO resolution.
  index_one(
    idx, contract_def, /*is_base=*/false, seen_fn_names, seen_mod_names);

  if (!ast || !ast->contains("nodes") || !ast->at("nodes").is_array())
    return idx;
  if (
    !contract_def.contains("linearizedBaseContracts") ||
    !contract_def.at("linearizedBaseContracts").is_array())
    return idx;

  // Resolve id → base contract body from the AST.
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
  // lin[0] is the target itself (already indexed).  Subsequent entries
  // are bases in MRO order.
  for (size_t i = 1; i < lin.size(); ++i)
  {
    if (!lin[i].is_number_integer())
      continue;
    int bid = lin[i].get<int>();
    auto it = contract_by_id.find(bid);
    if (it == contract_by_id.end())
      continue;
    index_one(
      idx, *it->second, /*is_base=*/true, seen_fn_names, seen_mod_names);
  }
  return idx;
}

/// Walker state passed by reference through the recursion.
struct Walker
{
  const ContractIndex &idx;
  RWSet rw;
  // callees discovered in this function body (as AST ids of callable defs).
  std::set<int> callees;
};

static void walk(Walker &w, const nlohmann::json &node, bool is_write_target);

static void
walk_children(Walker &w, const nlohmann::json &node, bool is_write_target)
{
  if (node.is_object())
  {
    for (auto it = node.begin(); it != node.end(); ++it)
      walk(w, it.value(), is_write_target);
  }
  else if (node.is_array())
  {
    for (const auto &child : node)
      walk(w, child, is_write_target);
  }
}

static void walk(Walker &w, const nlohmann::json &node, bool is_write_target)
{
  if (!node.is_object())
  {
    if (node.is_array())
      for (const auto &c : node)
        walk(w, c, is_write_target);
    return;
  }

  const std::string nt = node.value("nodeType", "");

  if (nt == "Identifier")
  {
    if (node.contains("referencedDeclaration"))
    {
      int ref = node["referencedDeclaration"].is_number_integer()
                  ? node["referencedDeclaration"].get<int>()
                  : -1;
      if (ref >= 0)
      {
        if (w.idx.state_var_ids.count(ref))
        {
          if (is_write_target)
            w.rw.writes.insert(ref);
          else
            w.rw.reads.insert(ref);
        }
        // Note: callees are recorded only via FunctionCall to avoid
        // counting function-typed assignments etc.
      }
    }
    return; // Identifier has no children to recurse into.
  }

  if (nt == "Assignment")
  {
    const std::string op = node.value("operator", "=");
    bool compound = (op != "=");
    if (node.contains("leftHandSide"))
      walk(w, node["leftHandSide"], true);
    if (compound && node.contains("leftHandSide"))
      // x += e  reads x as well — re-visit LHS as a read.
      walk(w, node["leftHandSide"], false);
    if (node.contains("rightHandSide"))
      walk(w, node["rightHandSide"], false);
    return;
  }

  if (nt == "UnaryOperation")
  {
    const std::string op = node.value("operator", "");
    bool writes_arg = (op == "++" || op == "--" || op == "delete");
    if (node.contains("subExpression"))
    {
      walk(w, node["subExpression"], writes_arg || is_write_target);
      if ((op == "++" || op == "--") && node.contains("subExpression"))
        walk(w, node["subExpression"], false); // read-then-write
    }
    return;
  }

  if (nt == "IndexAccess")
  {
    if (node.contains("baseExpression"))
      walk(w, node["baseExpression"], is_write_target);
    if (node.contains("indexExpression"))
      walk(w, node["indexExpression"], false);
    return;
  }

  if (nt == "MemberAccess")
  {
    // Detect `<addr>.balance` reads — record on the virtual __balance
    // token so candidate-pair analysis can flag balance-only TOD even
    // when no public state variable changes.
    const std::string mname = node.value("memberName", "");
    if (mname == "balance" && node.contains("expression"))
    {
      const auto &expr = node["expression"];
      const std::string et = expr.value("typeDescriptions", nlohmann::json{})
                               .value("typeString", "");
      if (et == "address" || et == "address payable")
        w.rw.reads.insert(kBalanceId);
    }
    if (node.contains("expression"))
      walk(w, node["expression"], is_write_target);
    return;
  }

  if (nt == "FunctionCallOptions")
  {
    // `addr.call{value: x}(args)` — record balance W on the virtual
    // __balance token.  The wrapped call expression / option values are
    // walked below so any state vars used as the value source are still
    // captured as reads.
    if (node.contains("names") && node["names"].is_array())
    {
      for (const auto &n : node["names"])
        if (n.is_string() && n.get<std::string>() == "value")
        {
          w.rw.writes.insert(kBalanceId);
          break;
        }
    }
    walk_children(w, node, false);
    return;
  }

  if (nt == "FunctionCall")
  {
    // Detect direct value-transferring builtins on an address-typed
    // base: `<addr>.transfer(v)` and `<addr>.send(v)` both decrement
    // `this->$balance` in the model, so flag W on __balance.
    if (node.contains("expression") && node["expression"].is_object())
    {
      const auto &callee = node["expression"];
      if (callee.value("nodeType", "") == "MemberAccess")
      {
        const std::string mname = callee.value("memberName", "");
        if (
          (mname == "transfer" || mname == "send") &&
          callee.contains("expression"))
        {
          const auto &base = callee["expression"];
          const std::string bt =
            base.value("typeDescriptions", nlohmann::json{})
              .value("typeString", "");
          if (bt == "address" || bt == "address payable")
            w.rw.writes.insert(kBalanceId);
        }
      }
    }

    // Direct internal call?  expression is Identifier whose
    // referencedDeclaration is one of our same-contract callables.
    if (node.contains("expression"))
    {
      const auto &expr = node["expression"];
      if (
        expr.is_object() && expr.value("nodeType", "") == "Identifier" &&
        expr.contains("referencedDeclaration"))
      {
        int ref = expr["referencedDeclaration"].is_number_integer()
                    ? expr["referencedDeclaration"].get<int>()
                    : -1;
        if (ref >= 0 && w.idx.all_callable_names.count(ref))
          w.callees.insert(ref);
      }
      walk(w, expr, false);
    }
    if (node.contains("arguments"))
      walk(w, node["arguments"], false);
    return;
  }

  if (nt == "ModifierInvocation")
  {
    if (node.contains("modifierName"))
    {
      const auto &mn = node["modifierName"];
      if (mn.is_object() && mn.contains("referencedDeclaration"))
      {
        int ref = mn["referencedDeclaration"].is_number_integer()
                    ? mn["referencedDeclaration"].get<int>()
                    : -1;
        if (ref >= 0 && w.idx.all_callable_names.count(ref))
          w.callees.insert(ref);
      }
    }
    if (node.contains("arguments"))
      walk(w, node["arguments"], false);
    return;
  }

  // Default: recurse into all children with the current write_target flag
  // disabled (most contexts are read contexts).
  walk_children(w, node, false);
}

/// Compute the per-callable footprint (R/W + outgoing call edges).
struct LocalInfo
{
  RWSet rw;
  std::set<int> callees;
};

static std::map<int, LocalInfo> compute_local(const ContractIndex &idx)
{
  std::map<int, LocalInfo> out;
  for (const auto &kv : idx.body_by_id)
  {
    int id = kv.first;
    const nlohmann::json &fn = *kv.second;
    Walker w{idx, {}, {}};
    if (fn.contains("body"))
      walk(w, fn["body"], false);
    // Modifier invocations attached to a FunctionDefinition live on the
    // FunctionDefinition itself, not in its body.
    if (fn.contains("modifiers"))
      for (const auto &mi : fn["modifiers"])
        walk(w, mi, false);
    // A `payable` function (or `receive` / `fallback` payable) implicitly
    // credits `this->$balance` with `msg.value` on entry — record W on
    // the virtual __balance token so candidate analysis pairs it up with
    // anything else that touches balance.
    if (fn.value("stateMutability", "") == "payable")
      w.rw.writes.insert(kBalanceId);
    out[id] = {w.rw, w.callees};
  }
  return out;
}

/// Iterate to a fixed point: footprint(f) = local(f) ∪ ⋃ footprint(callee).
static std::map<int, RWSet>
close_callgraph(const std::map<int, LocalInfo> &local)
{
  std::map<int, RWSet> closed;
  for (const auto &kv : local)
    closed[kv.first] = kv.second.rw;

  bool changed = true;
  while (changed)
  {
    changed = false;
    for (const auto &kv : local)
    {
      int id = kv.first;
      RWSet &dst = closed[id];
      for (int callee : kv.second.callees)
      {
        auto it = closed.find(callee);
        if (it == closed.end())
          continue;
        for (int r : it->second.reads)
          if (dst.reads.insert(r).second)
            changed = true;
        for (int wid : it->second.writes)
          if (dst.writes.insert(wid).second)
            changed = true;
      }
    }
  }
  return closed;
}

static bool is_externally_callable(const nlohmann::json &fn)
{
  const std::string vis = fn.value("visibility", "");
  return vis == "public" || vis == "external";
}

static bool is_orderable(const nlohmann::json &fn)
{
  // Reject things that can't appear in a TOD swap: ctors, fallback,
  // receive, view/pure (no writes).
  const std::string kind = fn.value("kind", "function");
  if (kind != "function")
    return false;
  const std::string mut = fn.value("stateMutability", "nonpayable");
  if (mut == "view" || mut == "pure")
    return false;
  return true;
}

} // namespace

std::map<std::string, RWSet>
compute_rw_sets(const nlohmann::json &contract_def, const nlohmann::json *ast)
{
  ContractIndex idx = index_contract(contract_def, ast);
  auto local = compute_local(idx);
  auto closed = close_callgraph(local);

  std::map<std::string, RWSet> out;
  for (const auto &kv : closed)
  {
    auto name_it = idx.all_callable_names.find(kv.first);
    if (name_it != idx.all_callable_names.end() && !name_it->second.empty())
      out[name_it->second] = kv.second;
  }
  return out;
}

std::vector<Pair> find_tod_candidates(
  const nlohmann::json &contract_def,
  Mode mode,
  const nlohmann::json *ast)
{
  ContractIndex idx = index_contract(contract_def, ast);
  auto local = compute_local(idx);
  auto closed = close_callgraph(local);

  // Collect orderable, externally callable function names + ids.
  std::vector<std::pair<std::string, int>> orderable;
  for (const auto &kv : idx.body_by_id)
  {
    const nlohmann::json &fn = *kv.second;
    if (fn.value("nodeType", "") != "FunctionDefinition")
      continue;
    if (!is_externally_callable(fn) || !is_orderable(fn))
      continue;
    const std::string name = fn.value("name", "");
    if (name.empty())
      continue;
    // Also require that the function actually has a non-empty footprint
    // (touches state in some way) — pure thunks with no writes would never
    // satisfy the candidacy check and would just clutter output.
    auto rw_it = closed.find(kv.first);
    if (rw_it == closed.end())
      continue;
    if (rw_it->second.writes.empty() && rw_it->second.reads.empty())
      continue;
    orderable.emplace_back(name, kv.first);
  }
  std::sort(orderable.begin(), orderable.end());

  std::vector<Pair> pairs;
  for (size_t i = 0; i < orderable.size(); ++i)
  {
    const RWSet &a = closed[orderable[i].second];
    for (size_t j = i + 1; j < orderable.size(); ++j)
    {
      const RWSet &b = closed[orderable[j].second];

      std::set<int> shared;
      // W(a) ∩ (R(b) ∪ W(b))
      for (int wid : a.writes)
      {
        if (b.reads.count(wid) || b.writes.count(wid))
          shared.insert(wid);
      }
      // W(b) ∩ (R(a) ∪ W(a))
      for (int wid : b.writes)
      {
        if (a.reads.count(wid) || a.writes.count(wid))
          shared.insert(wid);
      }
      if (shared.empty())
        continue;

      // Skip self-pairs (already excluded by j>i, but defensive).
      if (orderable[i].first == orderable[j].first)
        continue;

      const bool has_balance = shared.count(kBalanceId) > 0;
      const bool has_state_var = std::any_of(
        shared.begin(), shared.end(), [](int id) { return id > 0; });
      if (mode == Mode::BalanceOnly && !has_balance)
        continue;
      if (mode == Mode::RaceOnly && !has_state_var)
        continue;

      pairs.push_back(
        {orderable[i].first, orderable[j].first, std::move(shared)});
    }
  }
  return pairs;
}

} // namespace solidity_tod
