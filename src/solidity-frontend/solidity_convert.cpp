/// \file solidity_convert.cpp
/// \brief Top-level conversion driver and static member initialization.
///
/// Contains the solidity_convertert constructor, the main convert() entry
/// point that orchestrates the full AST-to-irep2 pipeline, and static member
/// initialization. The convert() method iterates over top-level AST nodes,
/// dispatching to contract, declaration, and utility conversion methods.

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
#include <regex>
#include <optional>

#include <fstream>

// initialize static members
const nlohmann::json solidity_convertert::empty_json = nlohmann::json::object();
std::string solidity_convertert::current_baseContractName = "";
nlohmann::json solidity_convertert::src_ast_json = empty_json;
std::unordered_map<std::string, typet> solidity_convertert::UserDefinedVarMap;

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
    is_pointer_check(true),
    nondet_bool_expr(),
    nondet_uint_expr(),
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

  set_sol_type(nondet_bool_expr.type(), SolidityGrammar::SolType::BOOL);
  set_sol_type(nondet_uint_expr.type(), SolidityGrammar::SolType::UINT256);

  addr_t = unsignedbv_typet(160);
  set_sol_type(addr_t, SolidityGrammar::SolType::ADDRESS);

  addrp_t = unsignedbv_typet(160);
  set_sol_type(addrp_t, SolidityGrammar::SolType::ADDRESS_PAYABLE);

  string_t = pointer_typet(signed_char_type());
  set_sol_type(string_t, SolidityGrammar::SolType::STRING);

  bool_t = bool_type();
  set_sol_type(bool_t, SolidityGrammar::SolType::BOOL);
  bool_t.set("#cpp_type", "bool");

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

// Convert smart contracts into symbol tables
bool solidity_convertert::convert()
{
  // merge the input files
  merge_multi_files();

  // By now the context should have the symbols of all ESBMC's intrinsics and the dummy main
  // We need to convert Solidity AST nodes to thstructe equivalent symbols and add them to the context
  // check if the file is suitable for verification
  if (contract_precheck())
    return true;

  absolute_path = src_ast_json["absolutePath"].get<std::string>();
  nlohmann::json &nodes = src_ast_json["nodes"];

  // store auxiliary info
  if (populate_auxiliary_vars())
    return true;

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
    bool found = false;
    auto it = funcSignatures.find(focus_cnt);
    if (it != funcSignatures.end())
    {
      for (const auto &m : it->second)
      {
        if (m.name != focus_func)
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
    {
      log_error(
        "--focus-function '{}' is not a public/external function of "
        "contract '{}'.",
        focus_func,
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
      // for constant variable defined in the file level which is outside the contract definition
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
  // this is to ensure that we have populated other auxiliary static variables before them
  for (const auto &c_name : contractNamesList)
    add_static_contract_instance(c_name);

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
  for (auto &ast_json : src_ast_json_array)
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

// topological sort is to make sure the order of contract AST is correct(Avoid some counterinstuitive cases)
// e.g. when contract A import B : contract A AST should be before contract B AST
void solidity_convertert::topological_sort(
  std::unordered_map<std::string, std::unordered_set<std::string>> &graph,
  std::unordered_map<std::string, nlohmann::json> &path_to_json,
  nlohmann::json &sorted_files)
{
  sorted_files.clear();
  std::unordered_map<std::string, int> in_degree;
  std::queue<std::string> zero_in_degree_queue;
  // Topological sorting function for sorting files according to import relationships
  // Calculate the in-degree for each node
  for (const auto &pair : graph)
  {
    if (in_degree.find(pair.first) == in_degree.end())
    {
      in_degree[pair.first] = 0;
    }
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
  while (!zero_in_degree_queue.empty())
  {
    std::string node = zero_in_degree_queue.front();
    zero_in_degree_queue.pop();
    // add the node's corresponding JSON file to the sorted result
    sorted_files.push_back(path_to_json[node]);
    // Update the in-degree of neighbouring nodes and add the new node with in-degree 0 to the queue
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
    log_error(
      "The minimum Solidity version ({}.{}.{}) < 0.5.0 is not supported. It is "
      "recommended "
      "to use a more recent Solidity version",
      min_version.major,
      min_version.minor,
      min_version.patch);
    return true;
  }
  else if (min_version >= v050 && min_version < v070)
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

      if (kind == "library")
        continue;
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
        auto c_def = find_decl_ref(inherit_id);
        assert(!c_def.empty());

        if (cname == c_def["name"].get<std::string>())
        {
          const std::string base_cname = j.first;
          inheritanceMap[cname].insert(base_cname);
          break;
        }
      }
    }
  }

  // setUp UserDefinedVarMap
  for (auto &t_node : nodes)
  {
    if (
      t_node.contains("nodeType") &&
      t_node["nodeType"] == "UserDefinedValueTypeDefinition")
    {
      typet t;
      if (get_type_description(t_node["underlyingType"]["typeDescriptions"], t))
        return true;
      std::string udv = t_node["name"].get<std::string>();
      UserDefinedVarMap[udv] = t;
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
  // since the based contract's signature is always coverred by the inherited one
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
    _sym.value = string;
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
    arr_t.set("#sol_array_size", std::to_string(length));

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
    sym.value = gen_zero(get_complete_type(_t, ns), true);
    sym.value.zero_initializer(true);

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
    fsym.value = fbody;
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
  is_init_sym.value = false_exprt();
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
  final_init_func_sym.value = init_func_body;

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

bool solidity_convertert::populate_low_level_functions(const std::string &cname)
{
  log_debug(
    "solidity",
    "Populating low-level function definition for contract {}",
    cname);

  exprt new_expr;
  // call("")
  if (get_call_definition(cname, new_expr))
    return true;
  move_builtin_to_contract(cname, new_expr, true);

  // call{}("")
  if (get_call_value_definition(cname, new_expr))
    return true;
  move_builtin_to_contract(cname, new_expr, true);

  // transfer()
  if (get_transfer_definition(cname, new_expr))
    return true;
  move_builtin_to_contract(cname, new_expr, true);

  // send()
  if (get_send_definition(cname, new_expr))
    return true;
  move_builtin_to_contract(cname, new_expr, true);

  // staticcall()
  if (get_staticcall_definition(cname, new_expr))
    return true;
  move_builtin_to_contract(cname, new_expr, true);

  // delegatecall()
  if (get_delegatecall_definition(cname, new_expr))
    return true;
  move_builtin_to_contract(cname, new_expr, true);

  return false;
}

/**
 * initialize the function signature set. Additionally, we merge inherited nodes.
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

      funcSignatures[cname].push_back(solidity_convertert::func_sig(
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
    funcSignatures[cname].push_back(solidity_convertert::func_sig(
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

  // After converting all AST nodes, current_functionDecl should be restored to nullptr.
  assert(current_functionDecl == nullptr);

  return false;
}
