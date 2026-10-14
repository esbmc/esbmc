/// \file solidity_convert.cpp
/// \brief Top-level conversion driver and static member initialization.
///
/// Contains the solidity_convertert constructor, the main convert() entry
/// point that orchestrates the full AST-to-irep2 pipeline, and static member
/// initialization. The convert() method iterates over top-level AST nodes,
/// dispatching to contract, declaration, and utility conversion methods.

#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <util/arith/arith_tools.h>
#include <util/arith/bitvector.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <util/config/focus_function.h>
#include <util/base/i2string.h>
#include <util/arith/mp_arith.h>
#include <util/irep/std_expr.h>
#include <util/message/message.h>
#include <limits>
#include <regex>
#include <optional>

#include <fstream>

// initialize static members
const nlohmann::json solidity_convertert::empty_json = nlohmann::json::object();
std::string solidity_convertert::current_baseContractName = "";
nlohmann::json solidity_convertert::src_ast_json = empty_json;
std::unordered_map<std::string, typet> solidity_convertert::UserDefinedVarMap;
std::unordered_map<std::string, const nlohmann::json *>
  solidity_convertert::fpc_memo;

solidity_convertert::solidity_convertert(
  contextt &_context,
  nlohmann::json &_ast_json,
  const std::string &_sol_cnts,
  const std::string &_sol_func,
  const std::string &_contract_path,
  const std::string &_focus_func)
  : context(_context),
    ns(context),
    src_ast_json_array(_ast_json),
    tgt_cnts(_sol_cnts),
    tgt_func(_sol_func),
    focus_func(_focus_func),
    contract_path(_contract_path),
    current_functionDecl(nullptr),
    current_forStmt(nullptr),
    expr_frontBlockDecl(code_blockt()),
    expr_backBlockDecl(code_blockt()),
    ctor_frontBlockDecl(code_blockt()),
    ctor_backBlockDecl(code_blockt()),
    current_lhsDecl(false),
    current_rhsDecl(false),
    current_functionName(""),
    member_entity_scope({}),
    initializers(code_blockt()),
    aux_counter(0),
    is_bound(false),
    is_reentry_check(false),
    is_reentry_balance_drain_check(false),
    outbound_drain_site_count(),
    is_pointer_check(true),
    nondet_bool_expr(),
    nondet_uint_expr(),
    nondet_uint256_expr(),
    nondet_bytes_dynamic_expr()
{
  std::ifstream in(_contract_path);
  contract_contents.assign(
    (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

  // bound setting - default value is false
  const std::string bound = config.options.get_option("bound");
  if (!bound.empty())
    is_bound = true;

  const std::string reentry_check = config.options.get_option("reentry-check");
  if (!reentry_check.empty())
    is_reentry_check = true;

  const std::string reentry_balance_drain_check =
    config.options.get_option("reentry-balance-drain-check");
  if (!reentry_balance_drain_check.empty())
    is_reentry_balance_drain_check = true;

  // solidity does not have pointer
  // however, in esbmc some array bounds check is related to the pointer check
  const std::string no_pointer_check =
    !config.options.get_option("no-pointer-check").empty()
      ? "1"
      : config.options.get_option("no-standard-checks");
  if (!no_pointer_check.empty())
    is_pointer_check = false;

  // initialize nondet_bool / nondet_uint
  if (
    context.find_symbol("c:@F@nondet_bool") == nullptr ||
    context.find_symbol("c:@F@nondet_uint") == nullptr)
  {
    log_error("Preprocessing error. Cannot find the NONDET symbol");
    abort();
  }
  if (context.find_symbol("c:@F@llc_nondet_bytes") == nullptr)
  {
    log_error("Preprocessing error. Cannot find the llc_nondet_bytes symbol");
    abort();
  }
  locationt l;
  get_library_function_call_no_args(
    "nondet_bool", "c:@F@nondet_bool", bool_t, l, nondet_bool_expr);
  get_library_function_call_no_args(
    "nondet_uint", "c:@F@nondet_uint", uint_type(), l, nondet_uint_expr);
  // 256-bit nondet for fields that legitimately span uint256 (balance,
  // codehash, code). Distinct from nondet_uint_expr so 32-bit uses
  // (e.g. switch case selectors) don't accidentally widen.
  get_library_function_call_no_args(
    "nondet_uint256",
    "c:@F@nondet_uint256",
    unsignedbv_typet(256),
    l,
    nondet_uint256_expr);

  set_sol_type(nondet_bool_expr.type(), SolidityGrammar::SolType::BOOL);
  set_sol_type(nondet_uint_expr.type(), SolidityGrammar::SolType::UINT256);
  set_sol_type(nondet_uint256_expr.type(), SolidityGrammar::SolType::UINT256);

  addr_t = unsignedbv_typet(160);
  set_sol_type(addr_t, SolidityGrammar::SolType::ADDRESS);

  addrp_t = unsignedbv_typet(160);
  set_sol_type(addrp_t, SolidityGrammar::SolType::ADDRESS_PAYABLE);

  string_t = pointer_typet(signed_char_type());
  set_sol_type(string_t, SolidityGrammar::SolType::STRING);

  bool_t = bool_type();
  set_sol_type(bool_t, SolidityGrammar::SolType::BOOL);
  bool_t.cpp_type("bool");

  byte_dynamic_t = symbol_typet(lib_prefix + "BytesDynamic");
  set_sol_type(byte_dynamic_t, SolidityGrammar::SolType::BYTES_DYN);

  // initialize nondet_bytes_dynamic_expr — used for LLC return data field
  get_library_function_call_no_args(
    "llc_nondet_bytes",
    "c:@F@llc_nondet_bytes",
    byte_dynamic_t,
    l,
    nondet_bytes_dynamic_expr);
  set_sol_type(
    nondet_bytes_dynamic_expr.type(), SolidityGrammar::SolType::BYTES_DYN);

  byte_static_t = symbol_typet(lib_prefix + "BytesStatic");
  set_sol_type(byte_static_t, SolidityGrammar::SolType::BYTES_STATIC);
}

// Recursively erase all JSON-null object fields. solc 0.6.x emits many
// optional fields (subdenomination, falseBody, body for interface methods,
// arguments for parent specifiers, overrides, length, contractScope,
// inner typeIdentifier/typeString on ElementaryTypeName, ...) as JSON null,
// whereas 0.8.x omits them. The whole frontend is sprinkled with
// `if (x.contains("k")) ... = x["k"]` which assumes "contains() implies
// non-null"; that assumption only holds for 0.8.x. Stripping nulls up
// front lets every existing `contains()` check behave identically across
// solc versions without spreading null guards through every accessor.
static void strip_nulls(nlohmann::json &n)
{
  if (n.is_object())
  {
    for (auto it = n.begin(); it != n.end();)
    {
      if (it.value().is_null())
        it = n.erase(it);
      else
      {
        strip_nulls(it.value());
        ++it;
      }
    }
  }
  else if (n.is_array())
  {
    for (auto &el : n)
      strip_nulls(el);
  }
}

// Convert smart contracts into symbol tables
bool solidity_convertert::convert()
{
  // Normalize 0.6.x AST quirks (see strip_nulls).
  for (auto &j : src_ast_json_array)
    strip_nulls(j);

  // merge the input files
  merge_multi_files();

  // By now the context should have the symbols of all ESBMC's intrinsics and
  // the dummy main We need to convert Solidity AST nodes to thstructe
  // equivalent symbols and add them to the context check if the file is
  // suitable for verification
  if (contract_precheck())
    return true;

  absolute_path = src_ast_json["absolutePath"].get<std::string>();

  // Revert-observation feature gate: enable the mark/clear injection and the
  // relaxed no-snapshot revert lowering only when the source references
  // `__ESBMC_reverted` (the user must declare a stub of that name to compile
  // under solc, so its mere presence is a reliable signal).  A single string
  // scan of the merged AST keeps non-using units byte-for-byte unchanged.
  // See docs/claude/solidity/revert-observation.md.
  // Foundry `vm.expectRevert(...)` reuses the same revert-observation machinery
  // (the next-call `assert(_ESBMC_sol_reverted_flag)` needs reverts to be
  // marked rather than path-pruned), so its presence also enables the gate.
  // `--solidity-path-coverage` also enables the gate: a require/revert failure
  // is one of the complete paths it enumerates, so that path must stay FEASIBLE
  // (marked + returned) instead of being pruned by the legacy
  // `__ESBMC_assume(false)`. The emitted `_ESBMC_sol_mark_revert()` is
  // additionally the only positive evidence that separates a reverting exit
  // from a plain early `return` — both of which otherwise lower to the very
  // same `IF <guard> THEN GOTO <END_FUNCTION>`. The mark/clear calls are tagged
  // `skipped` and live in a library file, so they add no coverage obligations.
  {
    const std::string ast_str = src_ast_json.dump();
    uses_revert_observation =
      ast_str.find("__ESBMC_reverted") != std::string::npos ||
      ast_str.find("expectRevert") != std::string::npos ||
      config.options.get_bool_option("solidity-path-coverage-enabled");
  }

  // AST rewrite: specialize internal function-pointer parameters whose
  // callback is statically known at the call site. Runs before symbol
  // registration so clones participate in the normal conversion pipeline.
  if (monomorphize_fn_ptr_params())
    return true;

  nlohmann::json &nodes = src_ast_json["nodes"];

  // store auxiliary info
  if (populate_auxiliary_vars())
    return true;

  // Fresh run: drop any find_parent_contract memo carried over from a
  // previous convert() (the map is static, may persist in-process).
  fpc_memo.clear();

  // --focus-function validation: must identify a single target contract.
  // If the source declares exactly one (non-library, non-interface) contract
  // and --contract was not provided, auto-select it; otherwise require
  // --contract to disambiguate.
  if (!focus_func.empty())
  {
    if (!tgt_func.empty())
    {
      log_error(
        "--focus-function is incompatible with --function; --function runs "
        "the named function in isolation with nondet state, while "
        "--focus-function keeps the full contract harness and restricts "
        "only the dispatch loop.");
      return true;
    }

    std::set<std::string> verifiable;
    for (const auto &cn : contractNamesList)
      if (nonContractNamesList.find(cn) == nonContractNamesList.end())
        verifiable.insert(cn);

    if (tgt_cnt_set.empty())
    {
      if (verifiable.size() != 1)
      {
        log_error(
          "--focus-function requires --contract when the source declares "
          "more than one contract (found {}). Specify which contract owns "
          "'{}' via --contract <name>.",
          verifiable.size(),
          focus_func);
        return true;
      }
      tgt_cnt_set.insert(*verifiable.begin());
    }
    else if (tgt_cnt_set.size() != 1)
    {
      log_error(
        "--focus-function requires exactly one --contract target, got {}.",
        tgt_cnt_set.size());
      return true;
    }

    const std::string &focus_cnt = *tgt_cnt_set.begin();

    // --focus-function names a SET (comma- or space-separated); see
    // util/config/focus_function.h for why the parsing lives there rather than
    // here. EVERY name is checked, and every name that matched nothing is
    // reported in ONE message. Reporting only the first would make a user with
    // a ten-function list fix one typo per run, and -- worse -- a list whose
    // first name is right and whose second is wrong would pass this check
    // entirely if the loop stopped at the first success.
    const std::vector<std::string> focus_names =
      focus_function_names(focus_func);
    if (focus_names.empty())
    {
      log_error(
        "--focus-function was given the value '{}', which names no function at "
        "all. Pass one or more public/external function names of contract "
        "'{}', separated by commas or spaces.",
        focus_func,
        focus_cnt);
      return true;
    }

    std::string missing;
    for (const auto &want : focus_names)
    {
      bool found = false;
      auto it = funcSignatures.find(focus_cnt);
      if (it != funcSignatures.end())
      {
        for (const auto &m : it->second)
        {
          if (m.name != want)
            continue;
          if (
            m.visibility != "public" && m.visibility != "external" &&
            config.options.get_option("no-visibility").empty())
            continue;
          if (m.name == focus_cnt)
            continue;
          if (m.name == "receive" || m.name == "fallback")
            continue;
          found = true;
          break;
        }
      }
      if (!found)
        missing += (missing.empty() ? "" : ", ") + ("'" + want + "'");
    }
    if (!missing.empty())
    {
      log_error(
        "--focus-function {} is not a public/external function of "
        "contract '{}'.",
        missing,
        focus_cnt);
      return true;
    }
  }

  // for coverage and trace simplification: update include_files
  auto add_unique = [](const std::string &file) {
    if (
      std::find(
        config.ansi_c.include_files.begin(),
        config.ansi_c.include_files.end(),
        file) == config.ansi_c.include_files.end())
    {
      config.ansi_c.include_files.push_back(file);
    }
  };
  add_unique(absolute_path);

  std::string old_path = absolute_path;

  // Pre-round: walk every interface and register its nested
  // struct / enum / error / event symbols up front. Libraries processed
  // later in round 1 can have function signatures referencing
  // `IFoo.Bar` types, and those references would otherwise be unresolved
  // (the interface itself is only fully processed in round 2). Without
  // this pass, a library returning an interface-nested struct crashes on
  // named-return declaration. Order-independent: the interface body is
  // not fully converted here, only type-child symbols are registered.
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    if (
      (*itr)["nodeType"] == "ContractDefinition" &&
      (*itr).contains("contractKind") &&
      (*itr)["contractKind"] == "interface" && (*itr).contains("nodes"))
    {
      std::string if_name = (*itr)["name"].get<std::string>();
      std::string old = current_baseContractName;
      current_baseContractName = if_name;
      for (auto &sub : (*itr)["nodes"])
      {
        const std::string nt = sub["nodeType"].get<std::string>();
        if (
          nt == "ErrorDefinition" || nt == "EventDefinition" ||
          nt == "StructDefinition" || nt == "EnumDefinition")
        {
          if (get_noncontract_defition(sub))
            return true;
        }
      }
      current_baseContractName = old;
    }
  }

  // first round: handle definitions that can be outside of the contract
  // including struct, enum, interface, event, error, library, constant...
  // noted that some can also be inside the contract, e.g. struct, enum...
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    if ((*itr).contains("absolutePath"))
    {
      // for "import" cases
      // we assume the merged file's nodes order is not messed up
      absolute_path = (*itr)["absolutePath"];
      add_unique(absolute_path);
    }

    if (get_noncontract_defition(*itr))
      return true;
    if (
      (*itr)["nodeType"].get<std::string>() == "VariableDeclaration" &&
      (*itr)["mutability"].get<std::string>() == "constant")
    {
      // for constant variable defined in the file level which is outside the
      // contract definition
      exprt dump;
      if (get_var_decl(*itr, dump))
        return true;
    }
  }
  absolute_path = old_path;

  // second round: handle contract definition
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    if ((*itr).contains("absolutePath"))
      absolute_path = (*itr)["absolutePath"];

    std::string node_type = (*itr)["nodeType"].get<std::string>();

    if (node_type == "ContractDefinition")
    {
      assert((*itr).contains("name"));
      std::string _name = (*itr)["name"].get<std::string>();
      if (get_contract_definition(_name))
        return true;
    }

    // reset
    reset_auxiliary_vars();
  }

  // add static instance
  // note that we populate the static instance in the end
  // this is to ensure that we have populated other auxiliary static variables
  // before them
  // Base contracts of the --contract target are registered but NOT
  // deployed.
  //
  // In Solidity, deploying `Derived` runs `Base`'s constructor *as part of*
  // `Derived`'s construction; it does not create a second, independent
  // `Base` instance.  ESBMC models a contract's mapping state variables as
  // one static-lifetime global per contract *type*
  // (`sol:@C@<Base>@<var>#<id>`), and the inherited constructor body that
  // `Derived`'s constructor invokes writes exactly those globals.  So
  // deploying `_ESBMC_Object_<Base>` in addition to `_ESBMC_Object_<Derived>`
  // executes the base constructor TWICE against the same storage.
  //
  // A base constructor that is not idempotent w.r.t. its own storage then
  // kills the entire verification path: OpenZeppelin's `MinterRole()` calls
  // `_addMinter(msg.sender)` -> `Roles.add`, whose
  // `require(!has(role, account))` lowers to `ASSUME !bearer[msg.sender]`.
  // The first deployment sets `bearer[msg.sender] = true`, so the second
  // execution assumes `false`.  Everything after it — the target's own
  // constructor, `_ESBMC_Main_<T>`, and the whole public-function
  // dispatcher — becomes infeasible, and coverage collapses to the
  // base-constructor prefix regardless of --unwind.
  // (Reproducer: Dataset/transracer_50/sources/Aavio/contract.sol with
  // --contract Aavio; minimal form in
  // papers_to_write/esbmc_cov_test_gen/measure/repro/d2_min_sharedmap.sol.)
  //
  // Trade-off: an undeployed singleton keeps its `$address` unconstrained
  // instead of a fresh unique one. That is closer to reality (the contract
  // was never deployed), but it does mean the address-dispatch ladder can
  // no longer rule it out by construction. Restricted to the single
  // `--contract` target case; whole-file mode verifies every contract in
  // turn and keeps deploying all of them (byte-identical behaviour).
  for (const auto &c_name : contractNamesList)
    add_static_contract_instance(c_name, !is_strict_base_of_target(c_name));

  // --reentry-balance-drain-check: emit one [approx] warning per
  // contract that the user opted into the check on but had no
  // outbound transfer/send/call{value:} call sites for the helper to
  // wrap.  (Pure-deposit / payable-only contracts and pure
  // computational contracts both fall into this bucket — neither
  // needs the assert.)  Skip non-tracked contracts (interfaces,
  // libraries) since the conversion path doesn't reach them anyway.
  if (is_reentry_balance_drain_check)
  {
    for (const auto &c_name : contractNamesList)
    {
      if (nonContractNamesList.count(c_name) != 0)
        continue;
      if (outbound_drain_site_count[c_name] == 0)
        log_warning(
          "[approx] --reentry-balance-drain-check skipped: {} has no "
          "outbound value-transfer call sites",
          c_name);
    }
  }

  // Emit the enclosing-debit helper NOW — after every contract's
  // `_ESBMC_Object_<C>` static instance is registered and ctor
  // resolved.  Library $transfer/$call#1/$send bodies parsed
  // earlier already reference the helper by its symbol id (forward
  // declaration); only the BODY needs all contract instances
  // visible.  Skip if no contracts are tracked (the helper would
  // have nothing to dispatch to, but library bodies still reference
  // it, so emit an empty stub).
  if (build_enclosing_debit_helper())
    return true;

  // Do Verification
  // single contract verification: where the option "--contract" is set.
  // multiple contracts verification: essentially verify the whole file.
  // single contract
  if (tgt_func.empty())
  {
    if (tgt_cnt_set.size() == 1)
    {
      // perform multi-transaction verification
      // by adding symbols to the "sol_main()" entry function
      if (multi_transaction_verification(*tgt_cnt_set.begin()))
        return true;
    }
    // multiple contract
    // either --contract unset, or --contract "C1 C2 ..."
    else
    {
      // for bounded cross-contract verification  (--bound)
      if (is_bound && multi_contract_verification_bound(tgt_cnt_set))
        return true;
      // for unbounded cross-contract verification (--unbound)
      else if (multi_contract_verification_unbound(tgt_cnt_set))
        return true;
    }
  }
  // else: verify the target function.

  return false; // 'false' indicates successful completion.
}

/*
e.g.

{
  "absolutePath": "contract_import2.sol",
  "id": 67,
  "nodes": [
  {
    "absolutePath": "contract_import2.sol",
    ""id": 67,
  }
  {
    "contractKind": "contract",
    "name": "A",
  }
  {
    "absolutePath": "contract_import.sol",
    "id": 56,
  }
  {
    "contractKind": "contract",
    "name": "B",
  }
}

*/
void solidity_convertert::merge_multi_files()
{
  // no imports
  if (src_ast_json_array.size() <= 1)
  {
    src_ast_json = src_ast_json_array[0];
    return;
  }
  // Import relationship diagram
  std::unordered_map<std::string, std::unordered_set<std::string>> import_graph;
  // Path to JSON object mapping
  std::unordered_map<std::string, nlohmann::json> path_to_json;
  // Constructing an import relationship diagram
  for (const auto &ast_json : src_ast_json_array)
  {
    std::string path = ast_json["absolutePath"];
    path_to_json[path] = ast_json;
    std::unordered_set<std::string> imports;
    // Extract the import path from the ImportDirective node.
    for (const auto &node : ast_json["nodes"])
    {
      if (node["nodeType"] == "ImportDirective")
      {
        std::string import_path = node["absolutePath"];
        imports.insert(import_path);
      }
    }
    import_graph[path] = imports;
  }

  // Perform topological sorting
  topological_sort(import_graph, path_to_json, src_ast_json_array);
  //  reversal
  //  contract B is A{}; contract A{};
  // =>
  //  contract A{}; contract B is A{};
  std::reverse(src_ast_json_array.begin(), src_ast_json_array.end());
  std::vector<nlohmann::json> nodes, paths;
  for (auto &ast_json : src_ast_json_array)
  {
    // store path node (SourceUnit)
    nlohmann::json dump = ast_json;
    dump.erase("nodes");
    paths.push_back(dump);

    // remove all the `import` statements
    auto &_nodes = ast_json["nodes"];
    for (auto it = _nodes.begin(); it != _nodes.end();)
    {
      if ((*it)["nodeType"] == "ImportDirective")
        it = _nodes.erase(it); // erase returns the next valid iterator
      else
        ++it;
    }
    nodes.push_back(_nodes);
  }

  src_ast_json = src_ast_json_array[0];
  auto &_nodes = src_ast_json["nodes"];

  // Insert stripped SourceUnit node at the front
  _nodes.insert(_nodes.begin(), paths[0]);
  for (std::size_t i = 1; i < src_ast_json_array.size(); i++)
  {
    _nodes.push_back(paths[i]); // first path
    for (const auto &node : nodes[i])
      _nodes.push_back(node); // then add each individual node inside the array
  }
}

// topological sort is to make sure the order of contract AST is correct(Avoid
// some counterinstuitive cases) e.g. when contract A import B : contract A AST
// should be before contract B AST
void solidity_convertert::topological_sort(
  std::unordered_map<std::string, std::unordered_set<std::string>> &graph,
  std::unordered_map<std::string, nlohmann::json> &path_to_json,
  nlohmann::json &sorted_files)
{
  sorted_files.clear();
  std::unordered_map<std::string, int> in_degree;
  std::queue<std::string> zero_in_degree_queue;
  // Topological sorting function for sorting files according to import
  // relationships Calculate the in-degree for each node
  for (const auto &pair : graph)
  {
    in_degree.try_emplace(pair.first, 0);
    for (const auto &neighbor : pair.second)
    {
      if (pair.first != neighbor)
      {
        // Ignore the case of importing itself.
        in_degree[neighbor]++;
      }
    }
  }

  // Find all the nodes with 0 entry and add them to the queue.
  for (const auto &pair : in_degree)
  {
    if (pair.second == 0)
    {
      zero_in_degree_queue.push(pair.first);
    }
  }
  // Process nodes in the queue
  std::unordered_set<std::string> visited;
  auto drain_queue = [&]() {
    while (!zero_in_degree_queue.empty())
    {
      std::string node = zero_in_degree_queue.front();
      zero_in_degree_queue.pop();
      if (!visited.insert(node).second)
        continue;
      // add the node's corresponding JSON file to the sorted result
      sorted_files.push_back(path_to_json[node]);
      // Update the in-degree of neighbouring nodes and add the new node with
      // in-degree 0 to the queue
      for (const auto &neighbor : graph[node])
      {
        if (node != neighbor)
        { // Ignore the case of importing itself.
          in_degree[neighbor]--;
          if (in_degree[neighbor] == 0)
          {
            zero_in_degree_queue.push(neighbor);
          }
        }
      }
    }
  };
  drain_queue();

  // Cycle handling: Kahn's algorithm leaves nodes that participate in an
  // import cycle stuck at in_degree > 0, silently dropping them from the
  // merged AST. Modern Solidity idioms routinely produce such cycles
  // (interface Foo holds a struct used by library Bar; library Bar returns
  // Foo's types → Foo ↔ Bar). Break the deadlock by repeatedly forcing the
  // remaining node with the lowest residual in_degree into the queue and
  // re-draining. The order within a cycle does not matter for later symbol
  // resolution: find_decl_ref walks all top-level nodes regardless.
  while (visited.size() < graph.size())
  {
    std::string pick;
    int best = std::numeric_limits<int>::max();
    for (const auto &pair : graph)
    {
      if (visited.count(pair.first))
        continue;
      int d = in_degree[pair.first];
      if (d < best)
      {
        best = d;
        pick = pair.first;
      }
    }
    if (pick.empty())
      break;
    in_degree[pick] = 0;
    zero_in_degree_queue.push(pick);
    drain_queue();
  }
}

// check if the programs is suitable for verificaiton
bool solidity_convertert::contract_precheck()
{
  // check json file contains AST nodes as Solidity might change
  if (!src_ast_json.contains("nodes"))
  {
    log_error("JSON file does not contain any AST nodes");
    return true;
  }

  // check json file contains AST nodes as Solidity might change
  if (!src_ast_json.contains("absolutePath"))
  {
    log_error("JSON file does not contain absolutePath");
    return true;
  }

  // check solc-version
  if (check_sol_ver())
    return true;

  nlohmann::json &nodes = src_ast_json["nodes"];

  bool found_contract_def = false;
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    // ignore the meta information and locate nodes in ContractDefinition
    std::string node_type = (*itr)["nodeType"].get<std::string>();
    if (node_type == "ContractDefinition") // contains AST nodes we need
    {
      if ((*itr)["contractKind"] == "library" && tgt_func.empty())
      {
        // Skip if it's a library and target function is empty
        // since a library cannot verify as a contract
        continue;
      }
      found_contract_def = true;
      break;
    }
  }
  if (!found_contract_def)
  {
    log_error("No verification targets(contracts) were found in the program.");
    return true;
  }
  return false;
}

bool solidity_convertert::check_sol_ver()
{
  struct versiont
  {
    int major = 0;
    int minor = 0;
    int patch = 0;

    bool operator<(const versiont &other) const
    {
      if (major != other.major)
        return major < other.major;
      if (minor != other.minor)
        return minor < other.minor;
      return patch < other.patch;
    }

    bool operator>(const versiont &other) const
    {
      return other < *this;
    }

    bool operator<=(const versiont &other) const
    {
      return !(other < *this);
    }

    bool operator>=(const versiont &other) const
    {
      return !(*this < other);
    }

    bool operator==(const versiont &other) const
    {
      return major == other.major && minor == other.minor &&
             patch == other.patch;
    }
  };

  bool found_pragma = false;
  std::optional<versiont> lower_bound;
  std::optional<versiont> upper_bound;

  if (!src_ast_json.contains("nodes") || !src_ast_json["nodes"].is_array())
  {
    log_error("Cannot find 'nodes' in AST.");
    return true;
  }

  auto parse_version =
    [](const std::string &version_str) -> std::optional<versiont> {
    std::regex ver_regex(R"((\d+)\.(\d+)\.(\d+))");
    std::smatch match;
    if (std::regex_match(version_str, match, ver_regex))
    {
      versiont result;
      result.major = std::stoi(match[1].str());
      result.minor = std::stoi(match[2].str());
      result.patch = std::stoi(match[3].str());
      return result;
    }
    return std::nullopt;
  };

  for (const auto &node : src_ast_json["nodes"])
  {
    if (node.contains("nodeType") && node["nodeType"] == "PragmaDirective")
    {
      found_pragma = true;

      if (node.contains("literals") && node["literals"].is_array())
      {
        std::vector<std::string> literals;
        for (const auto &lit : node["literals"])
        {
          if (lit.is_string())
          {
            literals.push_back(lit.get<std::string>());
          }
        }

        std::string current_op;

        for (size_t i = 0; i < literals.size(); ++i)
        {
          const std::string &token = literals[i];

          if (
            token == ">=" || token == ">" || token == "<=" || token == "<" ||
            token == "^")
          {
            current_op = token;
            continue;
          }

          for (size_t len = 1; len <= 3 && (i + len - 1) < literals.size();
               ++len)
          {
            std::string combined;
            for (size_t j = 0; j < len; ++j)
            {
              combined += literals[i + j];
            }

            auto ver_opt = parse_version(combined);
            if (ver_opt.has_value())
            {
              versiont ver = ver_opt.value();

              if (current_op == ">=" || current_op == "^" || current_op.empty())
              {
                if (!lower_bound.has_value() || ver > lower_bound.value())
                {
                  lower_bound = ver;
                }
              }
              else if (current_op == "<=" || current_op == "<")
              {
                if (!upper_bound.has_value() || ver < upper_bound.value())
                {
                  upper_bound = ver;
                }
              }

              i += len - 1;
              break;
            }
          }
        }
      }
    }
  }

  if (!found_pragma)
  {
    log_warning("Cannot find 'PragmaDirective' in AST.");
    return false;
  }

  if (!lower_bound.has_value())
  {
    log_warning("Cannot determine minimum Solidity version from pragma.");
    return false;
  }

  versiont min_version = lower_bound.value();
  versiont v050 = {0, 5, 0};
  versiont v070 = {0, 7, 0};

  if (min_version < v050)
  {
    // The pragma's *lower* bound is pre-0.5. That alone isn't a
    // hard failure: if the pragma also has an upper bound allowing
    // 0.5+ (e.g. `pragma solidity >=0.4.0 <0.9.0;`), solc has
    // already compiled it with a modern compiler and the AST is
    // fine to ingest. Only reject when the contract is *pinned*
    // to pre-0.5 (no upper bound ≥ 0.5).
    if (!upper_bound.has_value() || upper_bound.value() < v050)
    {
      log_error(
        "The minimum Solidity version ({}.{}.{}) < 0.5.0 is not supported. It "
        "is recommended to use a more recent Solidity version",
        min_version.major,
        min_version.minor,
        min_version.patch);
      return true;
    }
    log_warning(
      "The Solidity pragma lower bound ({}.{}.{}) is pre-0.5; accepting "
      "because the upper bound permits a modern compiler.",
      min_version.major,
      min_version.minor,
      min_version.patch);
  }
  else if (min_version < v070)
  {
    log_warning(
      "The minimum solidity version ({}.{}.{}) < 0.7.0 may cause unexpected "
      "behaviour. It is recommended to use a more recent Solidity version.",
      min_version.major,
      min_version.minor,
      min_version.patch);
  }

  return false;
}

bool solidity_convertert::populate_auxiliary_vars()
{
  nlohmann::json &nodes = src_ast_json["nodes"];

  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    std::string node_type = (*itr)["nodeType"].get<std::string>();

    if (node_type == "ContractDefinition") // rule source-unit
    {
      std::string c_name = (*itr)["name"].get<std::string>();
      std::string kind = (*itr)["contractKind"].get<std::string>();
      bool is_abstract = (*itr)["abstract"].get<bool>();
      if (kind == "interface" || kind == "library" || is_abstract)
        nonContractNamesList.insert(c_name);
      if (kind == "interface")
        interfaceNamesList.insert(c_name);
      if (kind == "library")
      {
        libraryNamesList.insert(c_name);
        continue;
      }
      auto c_id = (*itr)["id"].get<int>();

      // store contract name
      contractNamesMap.insert(std::pair<int, std::string>(c_id, c_name));
      if (
        std::find(contractNamesList.begin(), contractNamesList.end(), c_name) ==
        contractNamesList.end())
      {
        contractNamesList.push_back(c_name); // Only push if not found
      }

      // store linearizedBaseList: inherit from who?
      // this is esstinally the calling order of the constructor
      for (const auto &id : (*itr)["linearizedBaseContracts"].items())
      {
        int _id = id.value().get<int>();
        linearizedBaseList[c_name].push_back(_id);
      }
      if (linearizedBaseList[c_name].empty())
        return true;
    }
  }

  // verifying targets
  if (!tgt_cnts.empty())
  {
    std::istringstream iss(tgt_cnts);
    std::string tgt_cnt;
    while (iss >> tgt_cnt)
      tgt_cnt_set.insert(tgt_cnt);
  }

  // TODO: Optimise
  // inheritanceMap: who inherit from me?
  // contract unknown; contract test is unknown
  // inheritanceMap[unknown] = {unknown, test}
  // inheritanceMap[test] = {test}
  for (auto i : contractNamesMap)
  {
    std::string cname = i.second;
    // add itself
    inheritanceMap[cname].insert(cname);
    for (auto j : linearizedBaseList)
    {
      for (auto inherit_id : j.second)
      {
        std::string base_cname = j.first;

        auto c_def = find_decl_ref(inherit_id);
        assert(!c_def.empty());

        if (cname == c_def["name"].get<std::string>())
        {
          inheritanceMap[cname].insert(base_cname);
          break;
        }
      }
    }
  }

  // setUp UserDefinedVarMap
  //
  // UserDefinedValueTypeDefinition nodes can appear either at file level
  // (`type MyInt is int;`) or nested inside a contract/library/interface
  // (`contract C { type MyInt is int; ... }`). References to the former
  // use the bare name (`MyInt`) while contract-scoped UDVTs are referred
  // to via the fully-qualified form (`C.MyInt`) in `typeDescriptions`.
  // Register both keys for contract-scoped UDVTs so that
  //   - UserDefinedTypeName resolution in solidity_grammar.cpp (which
  //     keys by the reference's typeString, `C.MyInt`) succeeds, and
  //   - MemberAccess lookup in solidity_convert_ref.cpp (which keys by
  //     the identifier name used in `MyInt.wrap(...)`, just `MyInt`)
  //     also succeeds.
  auto register_udvt =
    [&](const nlohmann::json &def, const std::string &scope) -> bool {
    typet t;
    if (get_type_description(def["underlyingType"]["typeDescriptions"], t))
      return true;
    const std::string name = def["name"].get<std::string>();
    // Record the source UDVT name on the underlying type so the Foundry
    // coverage-test generator can render `Name.wrap(<literal>)` (a bare
    // underlying literal is not assignable to a UDVT parameter). Inert for
    // symex/solver — it is a `#sol_*` attribute only. Scope-qualified for a
    // contract-nested UDVT (referred to as `Scope.Name`).
    t.set("#sol_udvt_name", scope.empty() ? name : scope + "." + name);
    UserDefinedVarMap[name] = t;
    if (!scope.empty())
      UserDefinedVarMap[scope + "." + name] = t;
    return false;
  };

  for (auto &t_node : nodes)
  {
    if (!t_node.contains("nodeType"))
      continue;
    const std::string nt = t_node["nodeType"].get<std::string>();
    if (nt == "UserDefinedValueTypeDefinition")
    {
      if (register_udvt(t_node, /*scope=*/""))
        return true;
    }
    else if (
      (nt == "ContractDefinition") && t_node.contains("nodes") &&
      t_node.contains("name"))
    {
      const std::string cname = t_node["name"].get<std::string>();
      for (auto &inner : t_node["nodes"])
      {
        if (
          inner.contains("nodeType") &&
          inner["nodeType"] == "UserDefinedValueTypeDefinition")
        {
          if (register_udvt(inner, cname))
            return true;
        }
      }
    }
  }

  // From here, we might start to modify the original src_ast_json
  for (auto &c_node : nodes)
  {
    //? should we consider library?
    if (
      c_node.contains("nodeType") &&
      c_node["nodeType"] == "ContractDefinition" && c_node.contains("name"))
    {
      if (populate_function_signature(c_node, c_node["name"]))
        return true;
    }
  }

  // initial structureTypingMap based on the inheritanceMap,
  // since the based contract's signature is always coverred by the inherited
  // one
  structureTypingMap = inheritanceMap;

  log_debug("solidity", "Matching function signautre");
  for (const auto &derived : contractNamesList)
  {
    for (const auto &base : contractNamesList)
    {
      if (structureTypingMap[derived].count(base) > 0)
        continue;

      // if derived implements all base functions by name+type
      if (is_func_sig_cover(derived, base))
      {
        log_debug("solidity", "contract {} covers contract {}", derived, base);
        structureTypingMap[derived].insert(base);
      }
    }
  }

  // add contract name string
  // const char * Base = &"Base"[0];
  for (auto contract_name : contractNamesList)
  {
    exprt _cname_expr;
    std::string aux_cname, aux_cid;
    aux_cname = contract_name;
    aux_cid = "sol:@" + aux_cname;

    string_constantt string(contract_name);
    set_sol_type(string.type(), SolidityGrammar::SolType::STRING_LITERAL);
    typet ct = string_t;
    ct.cmt_constant(true);
    symbolt s;
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    get_default_symbol(
      s, debug_modulename, ct, aux_cname, aux_cid, locationt());
    s.lvalue = true;
    s.static_lifetime = true; // static
    symbolt &_sym = *move_symbol_to_context(s);
    solidity_gen_typecast(ns, string, ct);
    _sym.set_value(string);
  }

  /* populate _bind_cname_list
  const char* Base = "Base";
  const char* Bank_bind_cname_list[1];
  void initialize_bind_list()
  {
    Bank_bind_cname_list[0] = Base;
  }
  */
  for (auto _cname : contractNamesList)
  {
    std::unordered_set<std::string> cname_set;
    unsigned int length = 0;

    cname_set = structureTypingMap[_cname];
    length = cname_set.size();
    assert(!cname_set.empty());
    if (length > 1)
    {
      // remove non-contract
      for (auto non_cname : nonContractNamesList)
      {
        if (non_cname == _cname)
          // we don't remove itself
          continue;
        if (cname_set.count(non_cname) != 0)
          cname_set.erase(non_cname);
      }
      // update length
      length = cname_set.size();
    }

    exprt size_expr;
    size_expr = constant_exprt(
      integer2binary(length, bv_width(uint_type())),
      integer2string(length),
      uint_type());

    typet ct = string_t;
    ct.cmt_constant(true);
    array_typet arr_t(ct, size_expr);
    set_sol_type(arr_t, SolidityGrammar::SolType::ARRAY);
    set_sol_array_size(arr_t, std::to_string(length));

    std::string aux_name, aux_id;
    aux_name = "$" + _cname + "_bind_cname_list";
    aux_id = "sol:@C@" + _cname + "@" + aux_name;
    symbolt s;
    typet _t = arr_t;
    _t.cmt_constant(true);
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    get_default_symbol(s, debug_modulename, _t, aux_name, aux_id, locationt());
    s.file_local = true;
    s.static_lifetime = true;
    s.lvalue = true;
    symbolt &sym = *move_symbol_to_context(s);
    {
      exprt _v = gen_zero(get_complete_type(_t, ns), true);
      _v.zero_initializer(true);
      sym.set_value(std::move(_v));
    }

    // f: initialize_bind_list
    std::string fname, fid;
    get_bind_cname_func_name(_cname, fname, fid);
    symbolt fs;
    code_typet ft;
    ft.return_type() = empty_typet();
    ft.make_ellipsis();
    get_default_symbol(fs, debug_modulename, ft, fname, fid, locationt());
    s.file_local = true;
    symbolt &fsym = *move_symbol_to_context(fs);

    // fbody:
    // Bank_bind_cname_list[0] = Base;
    // Bank_bind_cname_list[1] = Derived;
    // ...
    code_blockt fbody;
    unsigned int i = 0;
    exprt arr = symbol_expr(sym);
    // add hide
    code_labelt label;
    label.set_label("__ESBMC_HIDE");
    label.code() = code_skipt();
    fbody.operands().push_back(label);

    for (auto str : cname_set)
    {
      // lhs
      exprt pos = constant_exprt(
        integer2binary(i, bv_width(uint_type())),
        integer2string(i),
        uint_type());
      exprt idx = index_exprt(arr, pos, ct);
      // rhs
      exprt cname_str;
      get_cname_expr(str, cname_str);
      solidity_gen_typecast(ns, cname_str, ct);
      // assign
      exprt ass_expr = side_effect_exprt("assign", ct);
      ass_expr.copy_to_operands(idx, cname_str);
      convert_expression_to_code(ass_expr);

      fbody.copy_to_operands(ass_expr);
      ++i;
    }
    fsym.set_value(fbody);
  }

  // pupulate a function call _sol_init_()
  // 1. add a static var bool is_init = fasle
  // 2. add body
  // void _sol_init_()
  // {
  //   __ESBMC_hide;
  //   if (!is_init)
  //   {
  //     initialize();
  //     initialize_$A_cname_bind_list() // get_bind_cname_func_name
  //     initialize_$B_cname_bind_list()
  //     ...
  //   }
  //   is_init = true; // prevent re-init
  // }

  // 1. add a static var bool is_init = false
  symbolt is_init_sym;
  typet bool_type = bool_t;
  std::string is_init_name = "is_init";
  std::string is_init_id = "sol:@is_init";
  std::string debug_modulename = get_modulename_from_path(absolute_path);

  get_default_symbol(
    is_init_sym,
    debug_modulename,
    bool_type,
    is_init_name,
    is_init_id,
    locationt());
  is_init_sym.lvalue = true;
  is_init_sym.file_local = true;
  is_init_sym.static_lifetime = true;
  is_init_sym.set_value(false_exprt());
  symbolt &final_is_init_sym = *move_symbol_to_context(is_init_sym);

  // 2. add body
  // void _sol_init_()
  symbolt init_func_sym;
  code_typet init_func_type;
  init_func_type.return_type() = empty_typet();
  init_func_type.make_ellipsis();

  std::string init_func_name = "_sol_init_";
  std::string init_func_id = "sol:@F@_sol_init_";
  get_default_symbol(
    init_func_sym,
    debug_modulename,
    init_func_type,
    init_func_name,
    init_func_id,
    locationt());
  init_func_sym.file_local = true;
  symbolt &final_init_func_sym = *move_symbol_to_context(init_func_sym);

  // Function body
  code_blockt init_func_body;

  // Add __ESBMC_HIDE label
  {
    code_labelt label;
    label.set_label("__ESBMC_HIDE");
    label.code() = code_skipt();
    init_func_body.copy_to_operands(label);
  }

  // if (!is_init)
  exprt is_init_expr = symbol_expr(final_is_init_sym);
  exprt not_is_init = not_exprt(is_init_expr);

  // then block
  code_blockt then_block;

  // initialize(); — using helper to populate call
  {
    side_effect_expr_function_callt call;
    get_library_function_call_no_args(
      "initialize", "c:@F@initialize", empty_typet(), locationt(), call);
    convert_expression_to_code(call);
    then_block.move_to_operands(call);
  }

  // initialize_$A_cname_bind_list(); ...
  for (const auto &contract_name : contractNamesList)
  {
    std::string fname, fid;
    get_bind_cname_func_name(contract_name, fname, fid);
    side_effect_expr_function_callt bind_call;
    get_library_function_call_no_args(
      fname, fid, empty_typet(), locationt(), bind_call);
    convert_expression_to_code(bind_call);
    then_block.move_to_operands(bind_call);
  }

  // is_init = true;
  {
    exprt true_expr = true_exprt();
    code_assignt assign_is_init(is_init_expr, true_expr);
    then_block.copy_to_operands(assign_is_init);
  }

  // wrap into codet "ifthenelse"
  codet if_expr("ifthenelse");
  if_expr.copy_to_operands(not_is_init, then_block);

  // add to function body
  init_func_body.move_to_operands(if_expr);

  // assign body to function symbol
  final_init_func_sym.set_value(init_func_body);

  // for mapping
  extract_new_contracts();

  return false;
}

void solidity_convertert::get_cname_expr(
  const std::string &cname,
  exprt &new_expr)
{
  new_expr = symbol_expr(*context.find_symbol("sol:@" + cname));
}

bool solidity_convertert::build_enclosing_debit_helper()
{
  const std::string helper_id = "c:@F@_ESBMC_enclosing_debit";
  if (context.find_symbol(helper_id) != nullptr)
    return false; // already emitted

  typet addr_t_local = unsignedbv_typet(160);
  typet val_t = unsignedbv_typet(256);
  typet void_ptr_t = pointer_typet(empty_typet());

  code_typet helper_ft;
  helper_ft.return_type() = empty_typet();
  code_typet::argumentt val_arg;
  val_arg.type() = val_t;
  val_arg.cmt_base_name("val");
  val_arg.cmt_identifier(helper_id + "::val");
  helper_ft.arguments().push_back(val_arg);

  std::string dbgmod = get_modulename_from_path(absolute_path);
  locationt loc;
  symbolt helper_sym;
  get_default_symbol(
    helper_sym, dbgmod, helper_ft, "_ESBMC_enclosing_debit", helper_id, loc);
  helper_sym.lvalue = true;
  helper_sym.file_local = true;
  symbolt &added_helper = *move_symbol_to_context(helper_sym);

  symbolt val_sym;
  get_default_symbol(val_sym, dbgmod, val_t, "val", helper_id + "::val", loc);
  val_sym.lvalue = true;
  val_sym.is_parameter = true;
  val_sym.file_local = true;
  move_symbol_to_context(val_sym);

  code_blockt body;
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  body.move_to_operands(label);

  exprt val_expr = symbol_expr(*context.find_symbol(helper_id + "::val"));
  exprt encl_this =
    symbol_expr(*context.find_symbol("c:@_ESBMC_enclosing_contract_this"));

  // Emit one `if (encl_this == (void*)&_ESBMC_Object_<C>) { ...$balance -= val;
  // }` per known contract.  No `else` chaining — each branch short- circuits on
  // its own equality check, and at most one branch fires at runtime (pointer
  // equality is exclusive).
  for (const auto &cn : contractNamesList)
  {
    if (nonContractNamesList.count(cn) != 0)
      continue; // skip libraries, interfaces, abstract contracts

    exprt static_ins;
    get_static_contract_instance_ref(cn, static_ins);

    // (void*)&_ESBMC_Object_<cn>
    exprt addr_of = exprt("address_of", pointer_typet(static_ins.type()));
    addr_of.copy_to_operands(static_ins);
    exprt casted = addr_of;
    solidity_gen_typecast(ns, casted, void_ptr_t);

    exprt eq = exprt("=", bool_t);
    eq.copy_to_operands(encl_this, casted);

    exprt target_balance = member_exprt(static_ins, "$balance", val_t);
    exprt sub_assign = side_effect_exprt("assign-", val_t);
    sub_assign.copy_to_operands(target_balance, val_expr);
    convert_expression_to_code(sub_assign);

    code_blockt then_block;
    then_block.move_to_operands(sub_assign);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(eq, then_block);
    body.move_to_operands(if_expr);
  }

  added_helper.set_value(body);
  added_helper.set_type(helper_ft);
  return false;
}

bool solidity_convertert::populate_low_level_functions(
  const std::string &cname,
  bool is_library)
{
  log_debug(
    "solidity",
    "Populating low-level function definition for {} {}",
    is_library ? "library" : "contract",
    cname);

  exprt new_expr;
  // call("")
  if (get_call_definition(cname, new_expr, is_library))
    return true;
  move_builtin_to_contract(cname, new_expr, true);

  // call{}("")
  if (get_call_value_definition(cname, new_expr, is_library))
    return true;
  move_builtin_to_contract(cname, new_expr, true);

  // transfer()
  if (get_transfer_definition(cname, new_expr, is_library))
    return true;
  move_builtin_to_contract(cname, new_expr, true);

  // send()
  if (get_send_definition(cname, new_expr, is_library))
    return true;
  move_builtin_to_contract(cname, new_expr, true);

  // staticcall()
  if (get_staticcall_definition(cname, new_expr, is_library))
    return true;
  move_builtin_to_contract(cname, new_expr, true);

  // delegatecall()
  if (get_delegatecall_definition(cname, new_expr, is_library))
    return true;
  move_builtin_to_contract(cname, new_expr, true);

  return false;
}

/**
 * initialize the function signature set. Additionally, we merge inherited
 * nodes.
 * @json: parsing contract json
 * @cname: parsing contract name
 */
bool solidity_convertert::populate_function_signature(
  nlohmann::json &json,
  const std::string &cname)
{
  log_debug(
    "solidity", "Setting up the function signatures for contract {}", cname);
  assert(json.contains("contractKind"));
  assert(json.contains("nodes"));

  bool is_library = json["contractKind"] == "library";

  // merge inherited nodes
  if (!is_library)
  {
    std::set<std::string> dump;
    merge_inheritance_ast(cname, json, dump);
  }

  std::string func_name, func_id, visibility;
  code_typet type;
  bool is_inherit, is_payable;
  // check if the contract is library

  for (const auto &func_node : json["nodes"])
  {
    if (
      func_node.contains("nodeType") &&
      (func_node["nodeType"] == "FunctionDefinition"))
    {
      if (
        func_node["name"] == "" && func_node.contains("kind") &&
        func_node["kind"] == "constructor")
        func_name = cname;
      else
        func_name =
          func_node["name"] == "" ? func_node["kind"] : func_node["name"];
      func_id = "sol:@C@" + cname + "@F@" + func_name + "#" +
                i2string(func_node["id"].get<int>());
      if (get_func_decl_ref_type(func_node, type))
        return true;

      assert(
        func_node.contains("visibility") &&
        func_node.contains("stateMutability"));

      visibility = func_node["visibility"];
      is_payable = func_node["stateMutability"] == "payable";
      is_inherit = func_node.contains("is_inherited");

      funcSignatures[cname].push_back(
        solidity_convertert::func_sig(
          func_name,
          func_id,
          visibility,
          type,
          is_payable,
          is_inherit,
          is_library));
    }
  }

  // check implicit ctor:
  bool hasConstructor = std::any_of(
    funcSignatures[cname].begin(),
    funcSignatures[cname].end(),
    [&cname](const solidity_convertert::func_sig &sig) {
      return sig.name == cname;
    });
  if (!hasConstructor && !is_library)
  {
    func_name = cname;
    func_id = get_implict_ctor_call_id(cname);
    visibility = "public";
    is_payable = false;
    type.return_type() = empty_typet();
    type.return_type().set("cpp_type", "void");
    is_inherit = false;
    funcSignatures[cname].push_back(
      solidity_convertert::func_sig(
        func_name,
        func_id,
        visibility,
        type,
        is_payable,
        is_inherit,
        is_library));
  }

  return false;
}

bool solidity_convertert::convert_ast_nodes(
  const nlohmann::json &contract_def,
  const std::string &cname)
{
  // parse constructor
  if (get_constructor(contract_def, cname))
    return true;

  // flag the ctor of abstract/interface/library contracts as non-instantiable
  // so the Foundry coverage-test generator never emits `new <Abstract>(...)`
  mark_ctor_instantiability(cname);

  size_t index = 0;
  nlohmann::json ast_nodes = contract_def["nodes"];
  for (nlohmann::json::iterator itr = ast_nodes.begin(); itr != ast_nodes.end();
       ++itr, ++index)
  {
    nlohmann::json ast_node = *itr;
    std::string node_name = "";
    if (ast_node.contains("name"))
    {
      node_name = ast_node["name"].get<std::string>();
      if (node_name.empty() && !ast_node["kind"].is_null())
        ast_node["kind"].get<std::string>();
    }

    std::string node_type = ast_node["nodeType"].get<std::string>();
    log_debug(
      "solidity",
      "@@ Converting node[{}]: contract={}, name={}, nodeType={} ...",
      index,
      cname,
      node_name,
      node_type);

    // handle non-functional declaration,
    // due to that the vars/struct might be mentioned in the constructor
    exprt dummy_decl;
    if (get_non_function_decl(ast_node, dummy_decl))
      return true;

    // then we handle function definition
    if (get_function_decl(ast_node))
      return true;
  }

  // After converting all AST nodes, current_functionDecl should be restored to
  // nullptr.
  assert(current_functionDecl == nullptr);

  return false;
}
