/// \file solidity_monomorphize.cpp
/// \brief AST pass: specialize internal function-pointer parameters.
///
/// Internal function types are lowered to opaque void* in the Solidity
/// frontend (see solidity_convert_type.cpp, Pointer case for
/// FunctionTypeName). Indirect calls through them return nondet, which
/// destroys precision on map/reduce/apply-style helpers such as
/// ArrayUtils.map(square) where the callback is statically known at each
/// call site.
///
/// This pass runs before symbol registration. It scans every FunctionCall
/// whose callee is a FunctionDefinition with FunctionTypeName parameters
/// and whose corresponding positional arguments are direct function
/// references (Identifier or MemberAccess with referencedDeclaration
/// pointing at a FunctionDefinition). For each unique (callee, callbacks)
/// tuple it clones the callee, strips the fn-ptr parameters, rewrites
/// every indirect call through those parameters to a direct call of the
/// callback, and inserts the clone into the caller's enclosing contract
/// (so the clone inherits the caller's this-pointer scope). The original
/// call site is redirected to the clone and the fn-ptr arguments are
/// dropped.
///
/// Scope: one level of specialization only (no fixed-point). Callers that
/// pass fn-ptrs through a local or state variable still fall back to the
/// nondet indirect-call path.

#include <solidity-frontend/solidity_convert.h>
#include <util/message/message.h>

#include <nlohmann/json.hpp>

#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace
{
// Fresh id base for cloned nodes. solc AST ids are small (well under 10^5
// even on large contracts); reserve 10^8+ for clones so remapped ids never
// collide with existing ones. Each clone gets its own 10k-sized block.
int mono_counter = 0;
int fresh_id_offset()
{
  return 100000000 + (mono_counter++) * 100000;
}

// Extract the referencedDeclaration of an argument that is expected to be
// a direct function reference. Returns -1 for anything that is not a bare
// Identifier or a MemberAccess with a numeric referencedDeclaration.
int extract_callback_ref(const nlohmann::json &arg)
{
  if (!arg.is_object())
    return -1;
  const std::string nt = arg.value("nodeType", "");
  if (nt != "Identifier" && nt != "MemberAccess")
    return -1;
  if (!arg.contains("referencedDeclaration"))
    return -1;
  if (!arg["referencedDeclaration"].is_number())
    return -1;
  return arg["referencedDeclaration"].get<int>();
}

bool is_fn_ptr_param(const nlohmann::json &param)
{
  return param.is_object() && param.contains("typeName") &&
         param["typeName"].is_object() &&
         param["typeName"].value("nodeType", "") == "FunctionTypeName";
}

// Post-order DFS so inner calls (children in a method chain such as
// `range(l).map(sq).reduce(sum)`) are rewritten before their enclosing
// calls. Pre-order would invalidate pointers to the inner FunctionCall
// when the outer call replaces its MemberAccess `expression` field.
void collect_function_call_paths(
  nlohmann::json &node,
  std::vector<nlohmann::json *> &out)
{
  if (node.is_object())
  {
    for (auto &kv : node.items())
      collect_function_call_paths(kv.value(), out);
    if (node.value("nodeType", "") == "FunctionCall")
      out.push_back(&node);
  }
  else if (node.is_array())
  {
    for (auto &it : node)
      collect_function_call_paths(it, out);
  }
}

void collect_local_ids(const nlohmann::json &node, std::set<int> &ids)
{
  if (node.is_object())
  {
    if (node.contains("id") && node["id"].is_number())
      ids.insert(node["id"].get<int>());
    for (const auto &kv : node.items())
      collect_local_ids(kv.value(), ids);
  }
  else if (node.is_array())
  {
    for (const auto &it : node)
      collect_local_ids(it, ids);
  }
}

void remap_ids(
  nlohmann::json &node,
  const std::set<int> &local_ids,
  int offset)
{
  if (node.is_object())
  {
    if (node.contains("id") && node["id"].is_number())
    {
      int v = node["id"].get<int>();
      if (local_ids.count(v))
        node["id"] = v + offset;
    }
    if (
      node.contains("referencedDeclaration") &&
      node["referencedDeclaration"].is_number())
    {
      int v = node["referencedDeclaration"].get<int>();
      if (local_ids.count(v))
        node["referencedDeclaration"] = v + offset;
    }
    if (node.contains("scope") && node["scope"].is_number())
    {
      int v = node["scope"].get<int>();
      if (local_ids.count(v))
        node["scope"] = v + offset;
    }
    for (auto &kv : node.items())
      remap_ids(kv.value(), local_ids, offset);
  }
  else if (node.is_array())
  {
    for (auto &it : node)
      remap_ids(it, local_ids, offset);
  }
}

// Check whether every usage of Identifier `target_id` inside `subtree` is
// at FunctionCall.expression position (i.e. the parameter is only ever
// called, never assigned/stored/passed through). Returns false if any
// non-call usage is found. We walk manually so we can distinguish the
// "expression" child of a FunctionCall from any other containing field.
bool param_used_only_for_calls(
  const nlohmann::json &subtree,
  int target_id)
{
  if (subtree.is_object())
  {
    // If this is a FunctionCall, its `expression` child is allowed to be
    // `target_id` — recurse into `expression` with special treatment
    // (skip the identifier check for that exact slot) and into every
    // other field with the normal check.
    if (subtree.value("nodeType", "") == "FunctionCall")
    {
      for (const auto &kv : subtree.items())
      {
        if (kv.key() == "expression")
        {
          // If the expression itself is the target identifier, that is an
          // OK call-site usage. Otherwise recurse normally.
          const auto &e = kv.value();
          if (
            e.is_object() && e.value("nodeType", "") == "Identifier" &&
            e.contains("referencedDeclaration") &&
            e["referencedDeclaration"].is_number() &&
            e["referencedDeclaration"].get<int>() == target_id)
            continue; // allowed
          if (!param_used_only_for_calls(e, target_id))
            return false;
        }
        else
        {
          if (!param_used_only_for_calls(kv.value(), target_id))
            return false;
        }
      }
      return true;
    }

    // Non-FunctionCall node: flag any Identifier whose
    // referencedDeclaration == target_id as a forbidden usage.
    if (
      subtree.value("nodeType", "") == "Identifier" &&
      subtree.contains("referencedDeclaration") &&
      subtree["referencedDeclaration"].is_number() &&
      subtree["referencedDeclaration"].get<int>() == target_id)
      return false;

    for (const auto &kv : subtree.items())
      if (!param_used_only_for_calls(kv.value(), target_id))
        return false;
    return true;
  }
  if (subtree.is_array())
  {
    for (const auto &it : subtree)
      if (!param_used_only_for_calls(it, target_id))
        return false;
  }
  return true;
}

// Replace every FunctionCall inside `subtree` whose callee is an Identifier
// with referencedDeclaration == from_id by a clone whose callee is the
// callback reference template. Preserves the original callee's id/src so
// parent-contract lookup and location reporting remain stable.
void rewrite_indirect_calls(
  nlohmann::json &subtree,
  int from_id,
  const nlohmann::json &cb_ref_template)
{
  if (subtree.is_object())
  {
    if (
      subtree.value("nodeType", "") == "FunctionCall" &&
      subtree.contains("expression") && subtree["expression"].is_object())
    {
      auto &e = subtree["expression"];
      if (
        e.value("nodeType", "") == "Identifier" &&
        e.contains("referencedDeclaration") &&
        e["referencedDeclaration"].is_number() &&
        e["referencedDeclaration"].get<int>() == from_id)
      {
        nlohmann::json new_callee = cb_ref_template;
        if (e.contains("id"))
          new_callee["id"] = e["id"];
        if (e.contains("src"))
          new_callee["src"] = e["src"];
        // Clear argumentTypes — the call site carries its own.
        e = new_callee;
      }
    }
    for (auto &kv : subtree.items())
      rewrite_indirect_calls(kv.value(), from_id, cb_ref_template);
  }
  else if (subtree.is_array())
  {
    for (auto &it : subtree)
      rewrite_indirect_calls(it, from_id, cb_ref_template);
  }
}

// Locate the ContractDefinition node whose `nodes` array contains `target`
// transitively. Returns nullptr if target is at source-unit level.
nlohmann::json *find_enclosing_contract_nodes(
  nlohmann::json &root,
  const nlohmann::json &target)
{
  if (!root.is_object() || !root.contains("nodes") || !root["nodes"].is_array())
    return nullptr;

  nlohmann::json *hit = nullptr;
  std::function<bool(nlohmann::json &)> walk;
  walk = [&](nlohmann::json &n) -> bool {
    if (hit)
      return true;
    if (n.is_object())
    {
      if (n == target)
        return true;
      for (auto &kv : n.items())
      {
        if (walk(kv.value()))
        {
          if (
            !hit && n.value("nodeType", "") == "ContractDefinition" &&
            n.contains("nodes") && n["nodes"].is_array())
          {
            hit = &n["nodes"];
          }
          return true;
        }
      }
    }
    else if (n.is_array())
    {
      for (auto &it : n)
      {
        if (walk(it))
          return true;
      }
    }
    return false;
  };
  walk(root);
  return hit;
}

} // namespace

bool solidity_convertert::monomorphize_fn_ptr_params()
{
  nlohmann::json &root = src_ast_json;
  if (!root.is_object() || !root.contains("nodes") || !root["nodes"].is_array())
    return false;

  // Cache (callee_id, cb_ids) -> clone_id so repeated call sites share one
  // specialization.
  std::map<std::tuple<int, std::vector<int>, int>, int> clone_cache;

  // Iterate calls repeatedly: each pass may leave new FunctionCall nodes in
  // the cloned body, but we do not visit clones for further specialization
  // (one-level scope). Collect once from the pre-pass tree.
  std::vector<nlohmann::json *> calls;
  collect_function_call_paths(root["nodes"], calls);

  for (auto *call_ptr : calls)
  {
    nlohmann::json &call = *call_ptr;
    if (!call.contains("expression") || !call["expression"].is_object())
      continue;
    nlohmann::json &callee_expr = call["expression"];
    if (
      !callee_expr.contains("referencedDeclaration") ||
      !callee_expr["referencedDeclaration"].is_number())
      continue;
    int callee_id = callee_expr["referencedDeclaration"].get<int>();
    if (callee_id <= 0)
      continue;

    const nlohmann::json &callee_def = find_node_by_id(root, callee_id);
    if (
      callee_def.empty() ||
      callee_def.value("nodeType", "") != "FunctionDefinition")
      continue;
    if (
      !callee_def.contains("parameters") ||
      !callee_def["parameters"].contains("parameters"))
      continue;
    const nlohmann::json &params = callee_def["parameters"]["parameters"];
    if (!params.is_array())
      continue;

    std::vector<int> fn_param_positions;
    std::vector<int> fn_param_ids;
    for (size_t i = 0; i < params.size(); i++)
    {
      if (is_fn_ptr_param(params[i]))
      {
        fn_param_positions.push_back((int)i);
        fn_param_ids.push_back(params[i].value("id", -1));
      }
    }
    if (fn_param_positions.empty())
      continue;
    for (int id : fn_param_ids)
      if (id < 0)
      {
        fn_param_positions.clear();
        break;
      }
    if (fn_param_positions.empty())
      continue;

    // Refuse to specialize if the callee uses any fn-ptr parameter for
    // something other than a direct call (assignment, struct construction,
    // storing into state, passing through to another function, etc.).
    // In those cases dropping the parameter would leave dangling references
    // in the clone body. The call falls back to the existing nondet
    // indirect-call path.
    bool all_call_only = true;
    if (callee_def.contains("body"))
    {
      for (int pid : fn_param_ids)
      {
        if (!param_used_only_for_calls(callee_def["body"], pid))
        {
          all_call_only = false;
          break;
        }
      }
    }
    if (!all_call_only)
      continue;

    if (!call.contains("arguments") || !call["arguments"].is_array())
      continue;
    const nlohmann::json &args = call["arguments"];
    // solc's `using for` leaves `arr.map(f)` as a MemberAccess call with one
    // fewer argument than the declaration — the receiver (`arr`) binds to
    // the first parameter implicitly. Accept args.size() == params.size()
    // for explicit calls and args.size() == params.size() - 1 for the
    // using-for form. Refuse anything else (inherited constructor calls,
    // partial application, etc.).
    int implicit_self = (int)params.size() - (int)args.size();
    if (implicit_self != 0 && implicit_self != 1)
      continue;
    // If the implicit-self parameter is itself a fn-ptr (pathological), we
    // cannot resolve it to a direct reference, so skip.
    if (implicit_self == 1 && is_fn_ptr_param(params[0]))
      continue;

    // Resolve every fn-ptr argument to a FunctionDefinition. If any one is
    // not a direct function reference, abandon this call — the remaining
    // indirect calls will fall back to nondet.
    std::vector<int> cb_ids;
    std::vector<nlohmann::json> cb_templates;
    bool resolvable = true;
    for (int pos : fn_param_positions)
    {
      int arg_pos = pos - implicit_self;
      if (arg_pos < 0 || arg_pos >= (int)args.size())
      {
        resolvable = false;
        break;
      }
      int cb = extract_callback_ref(args[arg_pos]);
      if (cb <= 0)
      {
        resolvable = false;
        break;
      }
      const nlohmann::json &cb_def = find_node_by_id(root, cb);
      if (
        cb_def.empty() ||
        cb_def.value("nodeType", "") != "FunctionDefinition")
      {
        resolvable = false;
        break;
      }
      cb_ids.push_back(cb);
      cb_templates.push_back(args[arg_pos]);
    }
    if (!resolvable)
      continue;

    // Determine insertion site: the caller's enclosing contract (so the
    // clone runs under the caller's this-pointer scope when invoking
    // contract-method callbacks).
    nlohmann::json *insert_nodes = find_enclosing_contract_nodes(root, call);
    // Cache key includes the caller's insertion scope identity so two
    // identical specializations living in different contracts don't alias.
    int scope_key =
      insert_nodes ? reinterpret_cast<std::intptr_t>(insert_nodes) & 0x7fffffff
                   : 0;
    auto key = std::make_tuple(callee_id, cb_ids, scope_key);

    int clone_id;
    std::string clone_name;
    auto cache_it = clone_cache.find(key);
    if (cache_it != clone_cache.end())
    {
      clone_id = cache_it->second;
      const nlohmann::json &existing = find_node_by_id(root, clone_id);
      if (existing.empty())
      {
        // Cache is stale — fall through and rebuild.
        clone_cache.erase(cache_it);
      }
      else
      {
        clone_name = existing.value("name", "");
      }
    }

    if (clone_name.empty())
    {
      nlohmann::json clone = callee_def;

      // Remap all local ids with a unique offset so inner references don't
      // collide with the original callee's ids and DFS-based parent lookups
      // return the correct enclosing contract.
      std::set<int> local_ids;
      collect_local_ids(clone, local_ids);
      int offset = fresh_id_offset();
      remap_ids(clone, local_ids, offset);

      clone_id = clone.value("id", 0);
      std::string orig_name = callee_def.value("name", "fn");
      clone_name = orig_name + "__mono_" + std::to_string(mono_counter);
      clone["name"] = clone_name;
      clone.erase("nameLocation");
      clone.erase("functionSelector");
      // Force the clone to not be a free function even if the original was,
      // so `kind` lookups route correctly based on its new scope. We keep
      // "function" for contract-hosted clones; for top-level clones we
      // set "freeFunction" below.
      if (clone.value("kind", "") == "freeFunction")
      {
        // Will stay freeFunction; that path is handled at the insertion
        // branch below.
      }
      else
      {
        clone["kind"] = "function";
      }

      // Rewrite indirect calls in the clone's body.
      if (clone.contains("body"))
      {
        for (size_t k = 0; k < fn_param_ids.size(); k++)
        {
          int remapped_param_id = fn_param_ids[k] + offset;
          rewrite_indirect_calls(
            clone["body"], remapped_param_id, cb_templates[k]);
        }
      }

      // Strip FunctionTypeName parameters from the clone's parameter list.
      nlohmann::json kept = nlohmann::json::array();
      std::set<int> drop_set(
        fn_param_positions.begin(), fn_param_positions.end());
      for (size_t i = 0; i < clone["parameters"]["parameters"].size(); i++)
        if (drop_set.find((int)i) == drop_set.end())
          kept.push_back(clone["parameters"]["parameters"][i]);
      clone["parameters"]["parameters"] = kept;

      // Insert into the caller's enclosing contract, or top level if the
      // caller is a free function. A top-level clone must be marked as a
      // freeFunction so it doesn't try to acquire a this-pointer during
      // conversion.
      if (insert_nodes != nullptr)
      {
        insert_nodes->push_back(clone);
      }
      else
      {
        clone["kind"] = "freeFunction";
        root["nodes"].push_back(clone);
      }
      clone_cache[key] = clone_id;
    }

    // Rewrite the original call site: replace the callee with an Identifier
    // pointing at the clone and drop the fn-ptr arguments. When the call
    // came in as a using-for member call (`arr.map(f)`), capture the
    // MemberAccess base first and prepend it as the explicit first argument
    // so the clone receives `self` as a normal parameter.
    nlohmann::json receiver = nlohmann::json::object();
    bool has_receiver = false;
    if (
      implicit_self == 1 && callee_expr.value("nodeType", "") == "MemberAccess" &&
      callee_expr.contains("expression"))
    {
      receiver = callee_expr["expression"];
      has_receiver = true;
    }

    nlohmann::json new_callee = nlohmann::json::object();
    new_callee["nodeType"] = "Identifier";
    new_callee["name"] = clone_name;
    new_callee["referencedDeclaration"] = clone_id;
    new_callee["overloadedDeclarations"] = nlohmann::json::array();
    new_callee["typeDescriptions"] = nlohmann::json::object();
    new_callee["typeDescriptions"]["typeIdentifier"] =
      "t_function_internal_nonpayable$_";
    new_callee["typeDescriptions"]["typeString"] = "function ()";
    if (callee_expr.contains("id"))
      new_callee["id"] = callee_expr["id"];
    if (callee_expr.contains("src"))
      new_callee["src"] = callee_expr["src"];
    callee_expr = new_callee;

    nlohmann::json kept_args = nlohmann::json::array();
    if (has_receiver)
      kept_args.push_back(receiver);
    std::set<int> drop_arg_pos;
    for (int pos : fn_param_positions)
      drop_arg_pos.insert(pos - implicit_self);
    for (size_t i = 0; i < call["arguments"].size(); i++)
      if (drop_arg_pos.find((int)i) == drop_arg_pos.end())
        kept_args.push_back(call["arguments"][i]);
    call["arguments"] = kept_args;

    log_debug(
      "solidity",
      "monomorphize: cloned callee {} as {} (id {})",
      callee_id,
      clone_name,
      clone_id);
  }

  return false;
}
