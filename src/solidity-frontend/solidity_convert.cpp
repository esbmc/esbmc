#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/solidity_template.h>
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
#include <iostream>

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
  const std::string &_contract_path)
  : context(_context),
    ns(context),
    src_ast_json_array(_ast_json),
    tgt_cnts(_sol_cnts),
    tgt_func(_sol_func),
    contract_path(_contract_path),
    current_functionDecl(nullptr),
    current_forStmt(nullptr),
    current_typeName(nullptr),
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
    nondet_uint_expr()
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

  // initialize nondet_bool
  if (
    context.find_symbol("c:@F@nondet_bool") == nullptr ||
    context.find_symbol("c:@F@nondet_uint") == nullptr)
  {
    log_error("Preprocessing error. Cannot find the NONDET symbol");
    abort();
  }
  locationt l;
  get_library_function_call_no_args(
    "nondet_bool", "c:@F@nondet_bool", bool_type(), l, nondet_bool_expr);
  get_library_function_call_no_args(
    "nondet_uint", "c:@F@nondet_uint", uint_type(), l, nondet_uint_expr);
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
  if (populate_auxilary_vars())
    return true;

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
      //TODO: skip pattern base check as it's not really valuable at the moment.
      // assert(itr->contains("nodes"));
      // auto pattern_check =
      //   std::make_unique<pattern_checker>((*itr)["nodes"], sol_func);
      // pattern_check->do_pattern_check();
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

bool solidity_convertert::populate_auxilary_vars()
{
  nlohmann::json &nodes = src_ast_json["nodes"];

  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    std::string node_type = (*itr)["nodeType"].get<std::string>();

    if (node_type == "ContractDefinition") // rule source-unit
    {
      std::string c_name = (*itr)["name"].get<std::string>();
      std::string kind = (*itr)["contractKind"].get<std::string>();
      if (kind == "library")
        continue;
      auto c_id = (*itr)["id"].get<int>();

      // store contract name
      contractNamesMap.insert(std::pair<int, std::string>(c_id, c_name));
      contractNamesList.insert(c_name);

      // store linearizedBaseList: inherit from who?
      // this is esstinally the calling order of the constructor
      for (const auto &id : (*itr)["linearizedBaseContracts"].items())
      {
        int _id = id.value().get<int>();
        linearizedBaseList[c_name].push_back(_id);
      }
      if (linearizedBaseList[c_name].empty())
        return true;

      // auto _json = (*itr)["nodes"];
      // functionSignature[c_name].insert(c_name); // constructor
      // for(nlohmann::json::iterator ittr;ittr != _json.end(); ++ittr)
      // {}
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

        auto c_def = find_decl_ref(src_ast_json["nodes"], inherit_id);
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
    typet ct = pointer_typet(signed_char_type());
    ct.cmt_constant(true);
    symbolt s;
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    get_default_symbol(
      s, debug_modulename, ct, aux_cname, aux_cid, locationt());
    s.lvalue = true;
    s.file_local = true;
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

    exprt size_expr;
    size_expr = constant_exprt(
      integer2binary(length, bv_width(uint_type())),
      integer2string(length),
      uint_type());

    typet ct = pointer_typet(signed_char_type());
    ct.cmt_constant(true);
    array_typet arr_t(ct, size_expr);
    arr_t.set("#sol_type", "ARRAY");
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
      exprt cname_str = symbol_expr(*context.find_symbol("sol:@" + str));
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

  // for mapping
  extract_new_contracts();

  // create a _ESBMC_sol_init_ class and object
  if (get_esbmc_sol_init())
    return true;

  return false;
}

/*
class _ESBMC_sol_init_
{
  public:
  _ESBMC_sol_init_()
  {
    initialize();
    $_bind_cname_list();
  }
}
*/
bool solidity_convertert::get_esbmc_sol_init()
{
  std::string id, name;
  struct_typet t = struct_typet();

  name = "_ESBMC_sol_init_";
  id = prefix + name;
  t.tag(name);
  symbolt s;
  std::string debug = get_modulename_from_path(absolute_path);
  get_default_symbol(s, debug, t, name, id, locationt());
  s.is_type = true;
  symbolt &added_sym = *move_symbol_to_context(s);

  // constructor
  std::string fname, fid;
  fname = "_ESBMC_sol_init_";
  fid = "sol:@C@_ESBMC_sol_init_@F@" + fname + "#";
  code_typet ft;
  typet tmp_rtn_type("constructor");
  ft.return_type() = tmp_rtn_type;
  ft.set("#member_name", fid);
  ft.set("#inlined", true);

  symbolt fs;
  get_default_symbol(fs, debug, ft, fname, fid, locationt());
  fs.lvalue = true;
  fs.file_local = true;
  symbolt &added_fs = *move_symbol_to_context(fs);

  get_function_this_pointer_param(fname, fid, debug, locationt(), ft);
  added_fs.type = ft;

  code_blockt _block;
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  _block.operands().push_back(label);
  if (move_initializer_to_main(_block))
    return true;
  added_fs.value = _block;

  exprt sym = symbol_expr(added_fs);

  // add component to struct
  struct_typet::componentt comp;
  // construct comp
  comp.type() = sym.type();
  comp.identifier(sym.identifier());
  comp.name(sym.name());
  comp.pretty_name(sym.name());
  comp.set_access("public");
  comp.id("symbol");
  to_struct_type(added_sym.type).methods().push_back(comp);

  // add static contract instance
  std::string ctor_ins_name, ctor_ins_id;
  get_static_contract_instance_name(name, ctor_ins_name, ctor_ins_id);

  locationt ctor_ins_loc;
  ctor_ins_loc.file(absolute_path);
  ctor_ins_loc.line(1);
  std::string ctor_ins_debug_modulename =
    get_modulename_from_path(absolute_path);
  typet ctor_ins_typet = symbol_typet(prefix + name);

  symbolt ctor_ins_symbol;
  get_default_symbol(
    ctor_ins_symbol,
    ctor_ins_debug_modulename,
    ctor_ins_typet,
    ctor_ins_name,
    ctor_ins_id,
    ctor_ins_loc);
  ctor_ins_symbol.lvalue = true;
  ctor_ins_symbol.is_extern = false;
  // the instance should be set as static
  ctor_ins_symbol.static_lifetime = true;
  ctor_ins_symbol.file_local = true;

  auto &added_ins_sym = *move_symbol_to_context(ctor_ins_symbol);

  exprt ctor;
  side_effect_expr_function_callt call;
  call.function() = sym;
  call.type() = symbol_typet(prefix + name);

  exprt this_object;
  get_new_object(symbol_typet(prefix + name), this_object);
  call.arguments().push_back(this_object);

  // set constructor
  call.set("constructor", 1);

  side_effect_exprt tmp_obj("temporary_object", call.type());
  codet code_expr("expression");
  code_expr.operands().push_back(call);
  tmp_obj.initializer(code_expr);
  call.swap(tmp_obj);
  ctor = call;

  added_ins_sym.value = ctor;

  return false;
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
  std::set<std::string> dump;
  if (!is_library)
    merge_inheritance_ast(cname, json, dump);

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

bool solidity_convertert::get_non_function_decl(
  const nlohmann::json &ast_node,
  exprt &new_expr)
{
  new_expr = code_skipt();

  if (!ast_node.contains("nodeType"))
  {
    log_error("Missing \'nodeType\' filed in ast_node");
    abort();
  }

  SolidityGrammar::ContractBodyElementT type =
    SolidityGrammar::get_contract_body_element_t(ast_node);

  log_debug(
    "solidity",
    "\t@@@ Expecting non-function definition, Got {}",
    SolidityGrammar::contract_body_element_to_str(type));

  // based on each element as in Solidty grammar "rule contract-body-element"
  switch (type)
  {
  case SolidityGrammar::ContractBodyElementT::VarDecl:
  {
    return get_var_decl(ast_node, new_expr); // rule state-variable-declaration
  }
  case SolidityGrammar::ContractBodyElementT::StructDef:
  {
    return get_struct_class(ast_node); // rule enum-definition
  }
  case SolidityGrammar::ContractBodyElementT::ModifierDef:
  case SolidityGrammar::ContractBodyElementT::FunctionDef:
  case SolidityGrammar::ContractBodyElementT::EnumDef:
  case SolidityGrammar::ContractBodyElementT::ErrorDef:
  case SolidityGrammar::ContractBodyElementT::EventDef:
  case SolidityGrammar::ContractBodyElementT::UsingForDef:
  {
    break;
  }
  default:
  {
    log_error("Unimplemented type in rule contract-body-element");
    return true;
  }
  }
  return false;
}

bool solidity_convertert::get_function_decl(const nlohmann::json &ast_node)
{
  if (!ast_node.contains("nodeType"))
  {
    log_error("Missing \'nodeType\' filed in ast_node");
    abort();
  }

  SolidityGrammar::ContractBodyElementT type =
    SolidityGrammar::get_contract_body_element_t(ast_node);

  log_debug(
    "solidity",
    "\t@@@ Expecting function definition, Got {}",
    SolidityGrammar::contract_body_element_to_str(type));

  // based on each element as in Solidty grammar "rule contract-body-element"
  switch (type)
  {
  case SolidityGrammar::ContractBodyElementT::FunctionDef:
  {
    return get_function_definition(ast_node); // rule function-definition
  }
  case SolidityGrammar::ContractBodyElementT::VarDecl:
  case SolidityGrammar::ContractBodyElementT::StructDef:
  case SolidityGrammar::ContractBodyElementT::EnumDef:
  case SolidityGrammar::ContractBodyElementT::ErrorDef:
  case SolidityGrammar::ContractBodyElementT::EventDef:
  case SolidityGrammar::ContractBodyElementT::UsingForDef:
  case SolidityGrammar::ContractBodyElementT::ModifierDef:
  {
    break;
  }
  default:
  {
    log_error("Unimplemented type in rule contract-body-element");
    return true;
  }
  }
  return false;
}

// push back a this pointer to the type
void solidity_convertert::get_function_this_pointer_param(
  const std::string &contract_name,
  const std::string &func_id,
  const std::string &debug_modulename,
  const locationt &location_begin,
  code_typet &type)
{
  log_debug("solidity", "\t@@@ getting function this pointer param");
  assert(!contract_name.empty());
  code_typet::argumentt this_param;
  std::string this_name = "this";
  //? do we need to drop the '#n' tail in func_id?
  std::string this_id = func_id + "#" + this_name;

  this_param.cmt_base_name(this_name);
  this_param.cmt_identifier(this_id);

  this_param.type() = gen_pointer_type(symbol_typet(prefix + contract_name));
  symbolt param_symbol;
  get_default_symbol(
    param_symbol,
    debug_modulename,
    this_param.type(),
    this_name,
    this_id,
    location_begin);
  param_symbol.lvalue = true;
  param_symbol.is_parameter = true;
  param_symbol.file_local = true;

  if (context.find_symbol(this_id) == nullptr)
  {
    context.move_symbol_to_context(param_symbol);
  }

  type.arguments().push_back(this_param);
}

bool solidity_convertert::get_var_decl(
  const nlohmann::json &ast_node,
  exprt &new_expr)
{
  return get_var_decl(ast_node, empty_json, new_expr);
}

// rule state-variable-declaration
// rule variable-declaration-statement
bool solidity_convertert::get_var_decl(
  const nlohmann::json &ast_node,
  const nlohmann::json &initialValue,
  exprt &new_expr)
{
  if (ast_node.is_null() || ast_node.empty())
  {
    new_expr = nil_exprt();
    return false;
  }

  std::string current_contractName;
  get_current_contract_name(ast_node, current_contractName);
  bool is_library = !current_contractName.empty() &&
                    contractNamesList.count(current_contractName) == 0;

  // For Solidity rule state-variable-declaration:
  // 1. populate typet
  typet t;
  // VariableDeclaration node contains both "typeName" and "typeDescriptions".
  // However, ExpressionStatement node just contains "typeDescriptions".
  // For consistensy, we use ["typeName"]["typeDescriptions"] as in state-variable-declaration
  // to improve the re-usability of get_type* function, when dealing with non-array var decls.
  // For array, do NOT use ["typeName"]. Otherwise, it will cause problem
  // when populating typet in get_cast

  const nlohmann::json *old_typeName = current_typeName;
  current_typeName = &ast_node["typeName"];
  if (get_type_description(ast_node["typeName"]["typeDescriptions"], t))
    return true;
  current_typeName = old_typeName;

  bool is_contract =
    t.get("#sol_type").as_string() == "CONTRACT" ? true : false;
  bool is_mapping = t.get("#sol_type").as_string() == "MAPPING" ? true : false;
  bool is_new_expr = newContractSet.count(current_contractName);
  if (is_new_expr)
  {
    // hack: check if it's unbound and the only verifying targets
    if (
      !is_bound && tgt_cnt_set.count(current_contractName) > 0 &&
      tgt_cnt_set.size() == 1)
      is_new_expr = false;
  }

  // for mapping: populate the element type
  if (is_mapping && !is_new_expr)
  {
    assert(t.is_array());
    const auto &val_type = ast_node["typeName"]["valueType"];
    typet val_t;
    if (get_type_description(val_type["typeDescriptions"], val_t))
      return true;
    t.subtype() = val_t;
  }

  // set const qualifier
  bool is_constant =
    ast_node.contains("mutability") && ast_node["mutability"] == "constant";
  if (is_constant)
    t.cmt_constant(true);

  // record the state info
  // this will be used to decide if the var will be converted to this->var
  // when parsing function body.
  bool is_state_var = ast_node["stateVariable"].get<bool>();
  t.set("#sol_state_var", std::to_string(is_state_var));

  bool is_inherited = ast_node.contains("is_inherited");

  // 2. populate id and name
  std::string name, id;
  //TODO: Omitted variable
  if (ast_node["name"].get<std::string>().empty())
    // Omitted variable
    get_aux_var(name, id);
  else
  {
    if (get_var_decl_name(ast_node, name, id))
      return true;
  }

  // if we have already populated the var symbol, we do not need to re-parse
  // however, we need to return the symbol info
  if (context.find_symbol(id) != nullptr)
  {
    log_debug("solidity", "Found parsed symbol, skip parsing");
    new_expr = symbol_expr(*context.find_symbol(id));
    return false;
  }

  // 3. populate location
  locationt location_begin;
  get_location_from_node(ast_node, location_begin);

  // 4. populate debug module name
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  // 5. set symbol attributes
  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, t, name, id, location_begin);

  symbol.lvalue = true;
  // static_lifetime: this means it's defined in the file level, not inside contract
  // special case for mapping, even if it's inside a contract
  symbol.static_lifetime = current_contractName.empty() ||
                           (is_mapping && !is_new_expr) ||
                           (is_library && is_constant);
  symbol.file_local = !symbol.static_lifetime;
  symbol.is_extern = false;

  // For state var decl, we look for "value".
  // For local var decl, we look for "initialValue"
  bool has_init = (ast_node.contains("value") || !initialValue.empty());
  bool set_init = has_init && !is_inherited;
  const nlohmann::json init_value =
    ast_node.contains("value") ? ast_node["value"] : initialValue;
  const nlohmann::json literal_type = ast_node["typeDescriptions"];
  if (!set_init && !(is_mapping && is_new_expr))
  {
    // for both state and non-state variables, set default value as zero
    symbol.value = gen_zero(get_complete_type(t, ns), true);
    symbol.value.zero_initializer(true);
  }

  // 6. add symbol into the context
  // just like clang-c-frontend, we have to add the symbol before converting the initial assignment
  symbolt &added_symbol = *move_symbol_to_context(symbol);
  code_declt decl(symbol_expr(added_symbol));

  // 7. populate init value if there is any
  // special handling for array/dynarray
  std::string t_sol_type = t.get("#sol_type").as_string();
  exprt val;
  if (t_sol_type == "ARRAY" || t_sol_type == "ARRAY_LITERAL")
  {
    /** 
      uint[2] z;            // uint *z = (uint *)calloc(2, sizeof(uint));
      
                            // uint tmp1[2] = {1,2}; // populated into sym tab, not a real statement
      uint[2] zz = [1,2];   // uint *zz = (uint *)_ESBMC_arrcpy(tmp1, 2, 2, sizeof(uint));

      uint[2] y = x;        // uint *zz = (uint *)_ESBMC_arrcpy(x, 2, 2, sizeof(uint));

      TODO: suport disorder:
      uint[2] y = x;
      uint[2] x = [1,2];
    **/

    // get size
    std::string arr_size = "0";
    if (!t.get("#sol_array_size").empty())
      arr_size = t.get("#sol_array_size").as_string();
    else if (t.has_subtype() && !t.subtype().get("#sol_array_size").empty())
      arr_size = t.subtype().get("#sol_array_size").as_string();
    else
    {
      log_error("cannot get the size of fixed array");
      return true;
    }
    exprt size_expr = constant_exprt(
      integer2binary(string2integer(arr_size), bv_width(uint_type())),
      arr_size,
      uint_type());

    // get sizeof
    exprt size_of_expr;
    get_size_of_expr(t.subtype(), size_of_expr);

    if (set_init)
    {
      if (get_init_expr(init_value, literal_type, t, val))
        return true;

      side_effect_expr_function_callt acpy_call;
      get_arrcpy_function_call(location_begin, acpy_call);
      acpy_call.arguments().push_back(val);
      acpy_call.arguments().push_back(size_expr);
      acpy_call.arguments().push_back(size_of_expr);
      // typecast
      solidity_gen_typecast(ns, acpy_call, t);
      // set as rvalue
      added_symbol.value = acpy_call;
      decl.operands().push_back(acpy_call);
    }
    else
    {
      // do calloc
      side_effect_expr_function_callt calc_call;
      get_calloc_function_call(location_begin, calc_call);
      calc_call.arguments().push_back(size_expr);
      calc_call.arguments().push_back(size_of_expr);
      // typecast
      solidity_gen_typecast(ns, calc_call, t);
      // set as rvalue
      added_symbol.value = calc_call;
      decl.operands().push_back(calc_call);
    }
    exprt func_call;
    store_update_dyn_array(symbol_expr(added_symbol), size_expr, func_call);

    if (is_state_var && !is_inherited)
    {
      // move to ctor initializer
      move_to_initializer(func_call);
    }
    else
      move_to_back_block(func_call);
  }
  else if (t_sol_type == "DYNARRAY" && set_init)
  {
    exprt val;
    if (get_init_expr(init_value, literal_type, t, val))
      return true;

    if (val.is_typecast() || val.type().get("#sol_type") == "NEW_ARRAY")
    {
      // uint[] zz = new uint(10);
      // uint[] zz = new uint(len);
      //=> uint* zz = (uint *)calloc(10, sizeof(uint));
      solidity_gen_typecast(ns, val, t);
      added_symbol.value = val;
      decl.operands().push_back(val);

      // get rhs size, e.g. 10
      nlohmann::json callee_arg_json = init_value["arguments"][0];
      exprt size_expr;
      const nlohmann::json literal_type = callee_arg_json["typeDescriptions"];
      if (get_expr(callee_arg_json, literal_type, size_expr))
        return true;

      // construct statement _ESBMC_store_array(zz, 10);
      exprt func_call;
      store_update_dyn_array(symbol_expr(added_symbol), size_expr, func_call);

      if (is_state_var && !is_inherited)
      {
        // move to ctor initializer
        move_to_initializer(func_call);
      }
      else
        move_to_back_block(func_call);
    }
    else if (val.is_symbol())
    {
      /** 
      uint[] zzz;           // uint* zzz; // will not reach here actually
                            // 
      uint[] zzzz = [1,2];  // memcpy(zzzz, tmp2, 2*sizeof(uint));
                            // uint* zzzzz = 0;
      uint[2] zzzzz = z;    // memcpy(zzzzz, z, 2*sizeof(uint));
                            // uint* zzzzz = 0;
      uint[] zzzzzz = zzz;  // memcpy(zzzzzz, zzz, zzz.size * sizeof(uint));

      Theoretically we can convert it to something like int *z = new int[2]{0,1};
      However, this feature seems to be not fully supported in current esbmc-cpp (v7.6.1)
    */
      // get size
      exprt size_expr;
      get_size_expr(val, size_expr);

      // get sizeof
      exprt size_of_expr;
      get_size_of_expr(t.subtype(), size_of_expr);

      side_effect_expr_function_callt acpy_call;
      get_arrcpy_function_call(location_begin, acpy_call);
      acpy_call.arguments().push_back(val);
      acpy_call.arguments().push_back(size_expr);
      acpy_call.arguments().push_back(size_of_expr);
      // typecast
      solidity_gen_typecast(ns, acpy_call, t);
      // set as rvalue
      added_symbol.value = acpy_call;
      decl.operands().push_back(acpy_call);

      // construct statement _ESBMC_store_array(zz, 10);
      exprt func_call;
      store_update_dyn_array(symbol_expr(added_symbol), size_expr, func_call);

      if (is_state_var && !is_inherited)
      {
        // move to ctor initializer
        move_to_initializer(func_call);
      }
      else
        move_to_back_block(func_call);
    }
    else
    {
      log_error("Unexpect initialization for dynamic array");
      log_debug("solidity", "{}", val);
      return true;
    }
  }
  // special handling for mapping
  else if (is_mapping && is_new_expr)
  {
    // mapping(string => uint) test;
    // 1. the contract that contains this mapping is also used in a new expression
    // => __attribute__((annotate("__ESBMC_inf_size"))) struct _ESBMC_Mapping _ESBMC_inf_test[];
    // => struct mapping_t test = {_ESBMC_inf_test, this.address};
    // 2.
    // => struct mapping_t_fast test = {_ESBMC_inf_test};
    // 1. construct static infinite array
    std::string arr_name, arr_id;
    get_mapping_inf_arr_name(current_contractName, name, arr_name, arr_id);
    symbolt arr_s;
    std::string mapping_struct_name = "_ESBMC_Mapping";

    if (context.find_symbol(prefix + mapping_struct_name) == nullptr)
    {
      log_error("failed to find _ESBMC_Mapping reference");
      return true;
    }

    typet arr_t = array_typet(
      symbol_typet(prefix + mapping_struct_name), exprt("infinity"));
    get_default_symbol(
      arr_s, debug_modulename, arr_t, arr_name, arr_id, location_begin);
    arr_s.static_lifetime = true;
    arr_s.file_local = true;
    arr_s.lvalue = true;
    auto &add_added_s = *move_symbol_to_context(arr_s);
    add_added_s.value = gen_zero(get_complete_type(arr_t, ns), true);

    // 2. construct mapping_t struct instance's value
    typet map_t;
    map_t = context.find_symbol(prefix + "mapping_t")->type;

    assert(map_t.is_struct());
    exprt inits = gen_zero(map_t);

    exprt op0 = symbol_expr(add_added_s);
    // array => &array[0]
    solidity_gen_typecast(
      ns, op0, to_struct_type(map_t).components().at(0).type());
    inits.op0() = op0;

    // address => this->
    exprt this_expr;
    if (current_functionDecl)
    {
      if (get_func_decl_this_ref(*current_functionDecl, this_expr))
        return true;
    }
    else
    {
      if (get_ctor_decl_this_ref(ast_node, this_expr))
        return true;
    }
    exprt addr_expr =
      member_exprt(this_expr, "$address", unsignedbv_typet(160));
    solidity_gen_typecast(
      ns, addr_expr, to_struct_type(map_t).components().at(1).type());
    inits.op1() = addr_expr;

    added_symbol.value = inits;
    decl.operands().push_back(inits);
  }
  // now we have rule out other special cases
  else if (set_init)
  {
    if (get_init_expr(init_value, literal_type, t, val))
      return true;
    added_symbol.value = val;
    decl.operands().push_back(val);
  }

  // store state variable, which will be initialized in the constructor
  // note that for the state variables that do not have initializer
  // we have already set it as zero value
  // For unintialized contract type, no need to move to the initializer
  if (
    is_state_var && !is_inherited && !(is_contract && !has_init) &&
    !(is_mapping && !is_new_expr))
    move_to_initializer(decl);

  decl.location() = location_begin;
  new_expr = decl;

  log_debug(
    "solidity", "Finish parsing symbol {}", added_symbol.name.as_string());
  return false;
}

// This function handles both contract and struct
// The contract can be regarded as the class in C++, converting to a struct
bool solidity_convertert::get_struct_class(const nlohmann::json &struct_def)
{
  // 1. populate name, id
  std::string id, name;
  struct_typet t = struct_typet();
  std::string cname;

  if (struct_def["nodeType"].get<std::string>() == "ContractDefinition")
  {
    name = struct_def["name"].get<std::string>();
    id = prefix + name;
    t.tag(name);
    cname = name;
  }
  else if (struct_def["nodeType"].get<std::string>() == "StructDefinition")
  {
    // ""tag-struct Struct_Name"
    name = struct_def["name"].get<std::string>();
    id = prefix + "struct " + struct_def["canonicalName"].get<std::string>();
    t.tag("struct " + name);

    // populate the member_entity_scope
    // this map is used to find reference when there is no decl_ref_id provided in the nodes
    // or replace the find_decl_ref in order to speed up
    int scp = struct_def["id"].get<int>();
    member_entity_scope.insert(std::pair<int, std::string>(scp, name));
  }
  else
  {
    log_error(
      "Got nodeType={}. Unsupported struct type",
      struct_def["nodeType"].get<std::string>());
    return true;
  }

  log_debug("solidity", "Parsing struct/contract class {}", name);

  // 2. Check if the symbol is already added to the context, do nothing if it is
  // already in the context.
  if (context.find_symbol(id) != nullptr)
    return false;

  // 3. populate location
  locationt location_begin;
  get_location_from_node(struct_def, location_begin);

  // 4. populate debug module name
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, t, name, id, location_begin);

  symbol.is_type = true;
  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // 5. populate fields(state var) and method(function)
  // We have to add fields before methods as the fields are likely to be used
  // in the methods
  nlohmann::json ast_nodes;
  if (struct_def.contains("nodes"))
    ast_nodes = struct_def["nodes"];
  else if (struct_def.contains("members"))
    ast_nodes = struct_def["members"];
  else
  {
    // Defining empty structs is disallowed.
    // Contracts can be empty
    log_warning("Empty contract.");
  }

  for (nlohmann::json::iterator itr = ast_nodes.begin(); itr != ast_nodes.end();
       ++itr)
  {
    SolidityGrammar::ContractBodyElementT type =
      SolidityGrammar::get_contract_body_element_t(*itr);

    log_debug(
      "solidity",
      "@@@ got ContractBodyElementT = {}",
      SolidityGrammar::contract_body_element_to_str(type));

    switch (type)
    {
    case SolidityGrammar::ContractBodyElementT::VarDecl:
    {
      // this can be both state and non-state variable
      if (get_struct_class_fields(*itr, t))
        return true;
      break;
    }
    case SolidityGrammar::ContractBodyElementT::FunctionDef:
    {
      if (get_struct_class_method(*itr, t))
        return true;
      break;
    }
    case SolidityGrammar::ContractBodyElementT::StructDef:
    {
      exprt tmp_expr;
      if (get_noncontract_decl_ref(*itr, tmp_expr))
        return true;

      struct_typet::componentt comp;
      comp.swap(tmp_expr);
      comp.id("component");
      comp.type().set("#member_name", t.tag());

      if (get_access_from_decl(*itr, comp))
        return true;
      t.components().push_back(comp);
      break;
    }
    case SolidityGrammar::ContractBodyElementT::EnumDef:
    case SolidityGrammar::ContractBodyElementT::UsingForDef:
    case SolidityGrammar::ContractBodyElementT::ModifierDef:
    {
      // skip as it do not need to be populated to the value of the struct
      break;
    }
    case SolidityGrammar::ContractBodyElementT::ErrorDef:
    case SolidityGrammar::ContractBodyElementT::EventDef:
    {
      exprt tmp_expr;
      if (get_noncontract_decl_ref(*itr, tmp_expr))
        return true;
      struct_typet::componentt comp;
      comp.swap(tmp_expr);

      if (comp.is_code() && to_code(comp).statement() == "skip")
        break;

      // set virtual / override
      if ((*itr).contains("virtual") && (*itr)["virtual"] == true)
        comp.set("#is_sol_virtual", true);
      else if ((*itr).contains("overrides"))
        comp.set("#is_sol_override", true);

      t.methods().push_back(comp);
      break;
    }
    default:
    {
      log_error("Unimplemented type in rule contract-body-element");
      return true;
    }
    }
  }

  t.location() = location_begin;
  added_symbol.type = t;

  return false;
}

// parse a contract definition
bool solidity_convertert::get_contract_definition(const std::string &c_name)
{
  // cache
  // this is due to that we might call this funciton to parse another contract B
  // when we are parsing contract A
  auto old_current_baseContractName = current_baseContractName;
  auto old_current_functionName = current_functionName;
  auto old_current_functionDecl = current_functionDecl;
  auto old_current_forStmt = current_forStmt;
  auto old_current_typeName = current_typeName;
  auto old_initializers = initializers;

  // reset
  reset_auxiliary_vars();

  nlohmann::json &nodes = src_ast_json["nodes"];
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    std::string node_type = (*itr)["nodeType"].get<std::string>();
    if (
      node_type == "ContractDefinition" &&
      (*itr)["name"] == c_name) // rule source-unit
    {
      if (
        (*itr).contains("contractKind") && (*itr)["contractKind"] == "library")
        // we paerse library in the get_noncontract_defition
        continue;

      log_debug("solidity", "Parsing Contract {}", c_name);

      // set based contract name
      current_baseContractName = c_name;

      // set baseContracts
      // this will be used in ctor initialization
      nlohmann::json *based_contracts = nullptr;
      if ((*itr).contains("baseContracts") && !(*itr)["baseContracts"].empty())
        based_contracts = &((*itr)["baseContracts"]);

      nlohmann::json &ast_nodes = (*itr)["nodes"];
      for (nlohmann::json::iterator ittr = ast_nodes.begin();
           ittr != ast_nodes.end();
           ++ittr)
      {
        // struct/error/event....
        if (get_noncontract_defition(*ittr))
          return true;
      }

      for (nlohmann::json::iterator ittr = ast_nodes.begin();
           ittr != ast_nodes.end();
           ++ittr)
      {
        if (get_noncontract_defition(*ittr))
          return true;
      }

      // add a struct symbol for each contract
      // e.g. contract Base => struct Base
      if (get_struct_class(*itr))
        return true;

      // add solidity built-in property like balance, codehash
      if (add_auxiliary_members(c_name))
        return true;

      // parse contract body
      if (convert_ast_nodes(*itr, c_name))
        return true;
      log_debug("solidity", "Finish parsing contract {}'s body", c_name);

      // for inheritance
      bool has_inherit_from = inheritanceMap[c_name].size() > 1;
      if (
        has_inherit_from &&
        move_initializer_to_ctor(based_contracts, c_name, true))
        return true;

      // initialize state variable
      if (move_initializer_to_ctor(based_contracts, c_name))
        return true;

      symbolt s = *context.find_symbol(prefix + c_name);
    }
  }

  // restore
  current_baseContractName = old_current_baseContractName;
  current_functionName = old_current_functionName;
  current_functionDecl = old_current_functionDecl;
  current_forStmt = old_current_forStmt;
  current_typeName = old_current_typeName;
  initializers = old_initializers;

  return false;
}

bool solidity_convertert::get_struct_class_fields(
  const nlohmann::json &ast_node,
  struct_typet &type)
{
  struct_typet::componentt comp;

  if (get_var_decl_ref(ast_node, false, comp))
    return true;

  if (comp.type().get("#sol_type") == "MAPPING" && comp.type().is_array())
  {
    //! hack: for the (non-nested) mapping from contract that is not used in a new expression
    // we convert it to a global static inifinity array
    // thuse we do not populate it into the struct symbol
    return false;
  }

  comp.id("component");
  // TODO: add bitfield
  // if (comp.type().get_bool("#extint"))
  // {
  //   typet t;
  //   if (get_type_description(ast_node["typeName"]["typeDescriptions"], t))
  //     return true;

  //   comp.type().set("#bitfield", true);
  //   comp.type().subtype() = t;
  //   comp.set_is_unnamed_bitfield(false);
  // }
  comp.type().set("#member_name", type.tag());

  if (get_access_from_decl(ast_node, comp))
    return true;
  type.components().push_back(comp);

  return false;
}

bool solidity_convertert::get_struct_class_method(
  const nlohmann::json &ast_node,
  struct_typet &type)
{
  struct_typet::componentt comp;
  if (get_func_decl_ref(ast_node, comp))
    return true;

  log_debug(
    "solidity", "\t\t@@@ populating method {}", comp.identifier().as_string());

  if (comp.is_code() && to_code(comp).statement() == "skip")
    return false;

  if (get_access_from_decl(ast_node, comp))
    return true;

  // set virtual / override
  if (ast_node.contains("virtual") && ast_node["virtual"] == true)
    comp.set("#is_sol_virtual", true);
  else if (ast_node.contains("overrides"))
    comp.set("#is_sol_override", true);

  type.methods().push_back(comp);
  return false;
}

bool solidity_convertert::get_noncontract_decl_ref(
  const nlohmann::json &decl,
  exprt &new_expr)
{
  log_debug(
    "solidity",
    "\tget_noncontract_decl_ref, got nodeType={}",
    decl["nodeType"].get<std::string>());
  if (decl["nodeType"] == "StructDefinition")
  {
    std::string id;
    id = prefix + "struct " + decl["canonicalName"].get<std::string>();

    if (context.find_symbol(id) == nullptr)
    {
      if (get_struct_class(decl))
        return true;
    }

    new_expr = symbol_expr(*context.find_symbol(id));
  }
  else if (decl["nodeType"] == "ErrorDefinition")
  {
    std::string name, id;
    get_error_definition_name(decl, name, id);

    if (context.find_symbol(id) == nullptr)
      return true;
    new_expr = symbol_expr(*context.find_symbol(id));
  }
  else if (decl["nodeType"] == "EventDefinition")
  {
    // treat event as a function definition
    if (get_func_decl_ref(decl, new_expr))
      return true;
  }
  else if (
    decl["nodeType"] == "ContractDefinition" &&
    decl["contractKind"] == "library")
  {
    new_expr = code_skipt();
    new_expr.type().set("#sol_type", "LIBRARY");
  }
  else
  {
    log_error("Internal parsing error");
    abort();
  }

  return false;
}

// definition of event/error/interface/struct/library/...
bool solidity_convertert::get_noncontract_defition(nlohmann::json &ast_node)
{
  std::string node_type = (ast_node)["nodeType"].get<std::string>();
  log_debug(
    "solidity", "@@@ Expecting non-contract definition, got {}", node_type);

  if (node_type == "StructDefinition")
  {
    if (get_struct_class(ast_node))
      return true;
  }
  else if (node_type == "EnumDefinition")
    // set the ["Value"] for each member inside enum
    add_enum_member_val(ast_node);
  else if (node_type == "ErrorDefinition")
  {
    add_empty_body_node(ast_node);
    if (get_error_definition(ast_node))
      return true;
  }
  else if (node_type == "EventDefinition")
  {
    add_empty_body_node(ast_node);
    if (get_function_definition(ast_node))
      return true;
  }
  else if (node_type == "ContractDefinition" && ast_node["abstract"] == true)
  {
    // for abstract contract
    add_empty_body_node(ast_node);
  }
  else if (
    node_type == "ContractDefinition" && ast_node["contractKind"] == "library")
  {
    // for library entity
    // a library is equivalent to a static class
    std::string lib_name = ast_node["name"].get<std::string>();

    // we treat library as a contract, but we do not populate it as struct/contract symbol
    // instead, we only populate the entity and functions
    std::string old = current_baseContractName;
    current_baseContractName = lib_name;
    if (get_struct_class(ast_node))
      return true;

    if (convert_ast_nodes(ast_node, lib_name))
      return true;

    current_baseContractName = old;
  }

  return false;
}

// add a "body" node to funcitons within interfacae && abstract && event
// the idea is to utilize the function-handling APIs.
void solidity_convertert::add_empty_body_node(nlohmann::json &ast_node)
{
  //? will this affect find_decl_ref?
  if (ast_node["nodeType"] == "EventDefinition")
  {
    // for event-definition
    if (!ast_node.contains("body"))

      ast_node["body"] = {
        {"nodeType", "Block"},
        {"statements", nlohmann::json::array()},
        {"src", ast_node["src"]}};
  }
  else if (ast_node["contractKind"] == "interface")
  {
    // For interface: functions have no body
    for (auto &subNode : ast_node["nodes"])
    {
      if (
        (subNode["nodeType"] == "FunctionDefinition") &&
        !subNode.contains("body"))
        subNode["body"] = {
          {"nodeType", "Block"},
          {"statements", nlohmann::json::array()},
          {"src", ast_node["src"]}};
    }
  }
  else if (ast_node["abstract"] == true)
  {
    // For abstract: functions may or may not have body
    for (auto &subNode : ast_node["nodes"])
    {
      if (
        (subNode["nodeType"] == "FunctionDefinition") &&
        !subNode.contains("body"))
        subNode["body"] = {
          {"nodeType", "Block"},
          {"statements", nlohmann::json::array()},
          {"src", ast_node["src"]}};
    }
  }
}

void solidity_convertert::add_enum_member_val(nlohmann::json &ast_node)
{
  /*
  "nodeType": "EnumDefinition",
  "members": 
    [
      {
          "id": 2,
          "name": "SMALL",
          "nameLocation": "66:5:0",
          "nodeType": "EnumValue",
          "src": "66:5:0",
          "Value": 0 => new added object
      },
      {
          "id": 3,
          "name": "MEDIUM",
          "nameLocation": "73:6:0",
          "nodeType": "EnumValue",
          "src": "73:6:0",
          "Value": 1  => new added object
      },
    ] */

  assert(ast_node["nodeType"] == "EnumDefinition");
  int idx = 0;
  nlohmann::json &members = ast_node["members"];
  for (nlohmann::json::iterator itr = members.begin(); itr != members.end();
       ++itr, ++idx)
  {
    if (!(*itr).contains("Value"))
      (*itr).push_back(
        nlohmann::json::object_t::value_type("Value", std::to_string(idx)));
  }
}

// covert the error_definition to a function
bool solidity_convertert::get_error_definition(const nlohmann::json &ast_node)
{
  // e.g.
  // error errmsg(int num1, uint num2, uint[2] addrs);
  //   to
  // function 'tag-erro errmsg@12'() { __ESBMC_assume(false);}

  const nlohmann::json *old_functionDecl = current_functionDecl;
  const std::string old_functionName = current_functionName;

  std::string cname;
  get_current_contract_name(ast_node, cname);

  // e.g. name: errmsg; id: sol:@errmsg#12
  std::string name, id;
  get_error_definition_name(ast_node, name, id);
  const int id_num = ast_node["id"].get<int>();

  if (context.find_symbol(id) != nullptr)
  {
    current_functionDecl = old_functionDecl;
    current_functionName = old_functionName;
    return false;
  }
  // update scope map
  member_entity_scope.insert(std::pair<int, std::string>(id_num, name));

  // just to pass the internal assertions
  current_functionName = name;
  current_functionDecl = &ast_node;

  // no return value
  code_typet type;
  typet e_type = empty_typet();
  e_type.set("cpp_type", "void");
  type.return_type() = e_type;

  locationt location_begin;
  get_location_from_node(ast_node, location_begin);
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, type, name, id, location_begin);
  symbol.lvalue = true;

  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // populate the params
  SolidityGrammar::ParameterListT params =
    SolidityGrammar::get_parameter_list_t(ast_node["parameters"]);
  if (params == SolidityGrammar::ParameterListT::EMPTY)
    type.make_ellipsis();
  else
  {
    for (const auto &decl : ast_node["parameters"]["parameters"].items())
    {
      const nlohmann::json &func_param_decl = decl.value();

      code_typet::argumentt param;
      if (get_function_params(func_param_decl, cname, param))
        return true;

      type.arguments().push_back(param);
    }
  }
  added_symbol.type = type;

  // construct a "__ESBMC_assume(false)" statement
  typet return_type = empty_typet();
  locationt loc;
  get_location_from_node(ast_node, loc);
  side_effect_expr_function_callt call;
  get_library_function_call_no_args(
    "__ESBMC_assume", "c:@F@__ESBMC_assume", return_type, loc, call);

  exprt arg = false_exprt();
  call.arguments().push_back(arg);
  convert_expression_to_code(call);

  // insert it to the body
  code_blockt body;
  body.operands().push_back(call);
  added_symbol.value = body;

  // restore
  current_functionDecl = old_functionDecl;
  current_functionName = old_functionName;

  return false;
}

// add ["is_inherited"] = true to node and all sub_node that contains an "id"
void solidity_convertert::add_inherit_label(
  nlohmann::json &node,
  const std::string &cname)
{
  // Add or update the "is_inherited" label in the current node
  if (node.is_object() && node.contains("id"))
  {
    node["current_contract"] = cname;
    node["is_inherited"] = true;
  }

  // Traverse through all sub-nodes
  for (auto &sub_node : node)
  {
    if (sub_node.is_object() && sub_node.contains("id"))
    {
      sub_node["current_contract"] = cname;
      sub_node["is_inherited"] = true;
    }

    if (sub_node.is_object() || sub_node.is_array())
      add_inherit_label(sub_node, cname);
  }
}

/*
  prefix:
    c_: current contract, we need to merged the inherited contract nodes to it 
    i_: inherited contract
*/
void solidity_convertert::merge_inheritance_ast(
  const std::string &c_name,
  nlohmann::json &c_node,
  std::set<std::string> &merged_list)
{
  log_debug("solidity", "@@@ Merging AST for contract {}", c_name);
  // we have merged this contract
  if (merged_list.count(c_name) > 0)
    return;

  if (linearizedBaseList[c_name].size() > 1)
  {
    // this means the contract is inherited from others
    // skip the first one as it's contract itself
    for (auto i_ptr = linearizedBaseList[c_name].begin() + 1;
         i_ptr != linearizedBaseList[c_name].end();
         i_ptr++)
    {
      std::string i_name = contractNamesMap[*i_ptr];
      if (linearizedBaseList[i_name].size() > 1)
      {
        if (merged_list.count(i_name) == 0)
        {
          merged_list.insert(i_name);
          merge_inheritance_ast(i_name, c_node, merged_list);
        }
        else
          // we have merged this contract
          continue;
      }

      const nlohmann::json &i_node =
        find_decl_ref_unique_id(src_ast_json["nodes"], *i_ptr);
      assert(!i_node.empty());

      // abstract contract
      if (!i_node.contains("nodes"))
        continue;

      // *@i: incoming node
      // *@c_i: current node
      for (auto i : i_node["nodes"])
      {
        // skip duplicate
        bool is_dubplicate = false;
        for (const auto &c_i : c_node["nodes"])
        {
          if (c_i.contains("id") && c_i["id"] == i["id"])
          {
            is_dubplicate = true;
            break;
          }
        }
        if (is_dubplicate)
          continue;

        // skip ctor
        if (i.contains("kind") && i["kind"].get<std::string>() == "constructor")
          continue;

        // for virtual/override function
        assert(i.contains("name"));
        std::string i_name = i["name"].get<std::string>() == ""
                               ? i["kind"].get<std::string>()
                               : i["name"].get<std::string>();
        assert(!i_name.empty());
        if (i.contains("nodeType") && i["nodeType"] == "FunctionDefinition")
        {
          //! receive/fallback can be inherited but cannot be override.
          // to avoid the name ambiguous/conflict
          // order: current_contract -> most base -> derived
          bool is_conflict = false;

          assert(c_node.contains("nodes"));
          for (auto &c_i : c_node["nodes"])
          {
            if (
              c_i.contains("kind") &&
              c_i["kind"].get<std::string>() == "constructor")
              continue;

            if (
              c_i.contains("nodeType") &&
              c_i["nodeType"] == "FunctionDefinition")
            {
              assert(c_i.contains("name"));
              std::string c_iname = c_i["name"].get<std::string>() == ""
                                      ? c_i["kind"].get<std::string>()
                                      : c_i["name"].get<std::string>();
              assert(!c_iname.empty());

              if (i_name == c_iname)
              {
                /*
                   A
                  / \
                 B   C
                  \ /
                   D
                  for cases above, there must be an override inside D if B and C both override A.
                */
                is_conflict = true;
                break;
              }
            }
          }
          if (is_conflict)
            continue;
        }

        // Here we have ruled out the special cases
        // so that we could merge the AST
        log_debug(
          "solidity",
          "\t@@@ Merging AST node {} to contract {}",
          i_name,
          c_name);
        // This is to distinguish it from the originals
        add_inherit_label(i, c_name);

        c_node["nodes"].push_back(i);
      }
    }
  }
}

// parse the explicit ctor, or add the implicit ctor
bool solidity_convertert::get_constructor(
  const nlohmann::json &ast_node,
  const std::string &contract_name)
{
  log_debug("solidity", "Parsing Constructor {}...", contract_name);

  // check if we could find a explicit constructor
  nlohmann::json ast_nodes = ast_node["nodes"];
  for (nlohmann::json::iterator itr = ast_nodes.begin(); itr != ast_nodes.end();
       ++itr)
  {
    nlohmann::json ast_node = *itr;
    SolidityGrammar::ContractBodyElementT type =
      SolidityGrammar::get_contract_body_element_t(ast_node);
    switch (type)
    {
    case SolidityGrammar::ContractBodyElementT::FunctionDef:
    {
      if (
        ast_node.contains("kind") && !ast_node["kind"].empty() &&
        ast_node["kind"].get<std::string>() == "constructor")
        return get_function_definition(ast_node);
      continue;
    }
    default:
    {
      continue;
    }
    }
  }

  // reset
  assert(current_functionDecl == nullptr);

  // check if we need to add implicit constructor
  if (add_implicit_constructor(contract_name))
  {
    log_error("Failed to add implicit constructor");
    return true;
  }

  return false;
}

// add a empty constructor to the contract
bool solidity_convertert::add_implicit_constructor(
  const std::string &contract_name)
{
  log_debug("solidity", "\t@@@ Adding implicit constructor");
  std::string name, id;
  name = contract_name;

  // do nothing if there is already an explicit or implicit ctor
  get_ctor_call_id(contract_name, id);
  if (context.find_symbol(id) != nullptr)
    return false;

  // if we reach here, the id must be equal to get_implicit_ctor_id()
  // an implicit constructor is an void empty function
  code_typet type;
  type.return_type() = empty_typet();
  type.return_type().set("cpp_type", "void");

  locationt location_begin;

  std::string debug_modulename = get_modulename_from_path(absolute_path);

  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, type, name, id, location_begin);

  symbol.lvalue = true;
  symbol.is_extern = false;
  symbol.file_local = false;

  auto &sym = *move_symbol_to_context(symbol);

  code_blockt body_exprt = code_blockt();
  sym.value = body_exprt;

  // add this pointer
  get_function_this_pointer_param(
    contract_name, id, debug_modulename, location_begin, type);

  sym.type = type;
  return false;
}

void solidity_convertert::get_temporary_object(exprt &call, exprt &new_expr)
{
  side_effect_exprt tmp_obj("temporary_object", call.type());
  codet code_expr("expression");
  code_expr.operands().push_back(call);
  tmp_obj.initializer(code_expr);
  tmp_obj.location() = call.location();
  call.swap(tmp_obj);
  new_expr = call;
}

void solidity_convertert::convert_unboundcall_nondet(
  exprt &new_expr,
  const typet common_type,
  const locationt &l)
{
  if (
    new_expr.is_code() && new_expr.statement() == "function_call" &&
    new_expr.operands().size() >= 1 &&
    new_expr.op1().name() == "_ESBMC_Nondet_Extcall")
  {
    move_to_front_block(new_expr);
    get_nondet_expr(common_type, new_expr);
    new_expr.location() = l;
  }
}

/* the member access is something like:
  class Base
  {
  public:
    static Base instance;
    void test()
    {
      instance.doSomething(); // 
    }
    void doSomething()
    {
      assert(0);
    }
  };
  Base Base::instance;

  int main()
  {
    Base::instance.test();
  }
*/
bool solidity_convertert::get_unbound_expr(
  const nlohmann::json expr,
  const std::string &c_name,
  exprt &new_expr)
{
  log_debug("solidity", "get_unbound_expr");
  if (c_name.empty())
  {
    log_error("got empty contract name");
    return true;
  }
  // it's not a member access, as it can only jump within current contract
  assert(!c_name.empty());
  code_function_callt func_call;
  if (get_unbound_funccall(c_name, func_call))
    return true;

  locationt l;
  get_location_from_node(expr, l);
  func_call.location() = l;

  // reentry check
  if (is_reentry_check)
  {
    exprt this_expr;
    assert(current_functionDecl);
    if (get_func_decl_this_ref(*current_functionDecl, this_expr))
    {
      log_error("cannot get internal this pointer reference");
      return true;
    }

    exprt _mutex;
    get_contract_mutex_expr(c_name, this_expr, _mutex);

    exprt assign_lock = side_effect_exprt("assign", bool_type());
    assign_lock.copy_to_operands(_mutex, true_exprt());
    convert_expression_to_code(assign_lock);

    exprt assign_unlock = side_effect_exprt("assign", bool_type());
    assign_unlock.copy_to_operands(_mutex, false_exprt());
    convert_expression_to_code(assign_unlock);

    // this should before the unbound_func_call
    move_to_front_block(assign_lock);
    move_to_back_block(assign_unlock);
  }

  move_to_front_block(func_call);
  new_expr = func_call;
  return false;
}

// construct the unbound verification harness
bool solidity_convertert::get_unbound_function(
  const std::string &c_name,
  symbolt &sym)
{
  std::string h_name = "_ESBMC_Nondet_Extcall_" + c_name;
  std::string h_id = "sol:@C@" + c_name + "@" + h_name + "#";
  log_debug("solidity", "\tget_unbound_function {}", h_name);

  symbolt h_sym;

  if (context.find_symbol(h_id) != nullptr)
    h_sym = *context.find_symbol(h_id);
  else
  {
    // construct unbound_function

    // 1.0 func body
    code_blockt func_body;
    func_body.make_block();

    // add __ESBMC_HIDE
    code_labelt label;
    label.set_label("__ESBMC_HIDE");
    label.code() = code_skipt();
    func_body.operands().push_back(label);

    // 1.1 get static contract instance
    exprt contract_var;
    get_static_contract_instance_ref(c_name, contract_var);

    // construct return; to avoid fall-through
    exprt return_expr = code_returnt();

    // 2.0 check visibility setting
    bool skip_vis =
      config.options.get_option("no-visibility").empty() ? false : true;
    if (skip_vis)
    {
      log_warning(
        "force to verify every function, even it's an unreachable "
        "internal/private function. This might lead to false positives.");
    }

    // 2.1 construct if-then-else statement
    const auto methods = funcSignatures[c_name];

    for (const auto &method : methods)
    {
      // we only handle public (and external) function
      // as the private and internal function cannot be directly called
      if (
        !skip_vis && method.visibility != "public" &&
        method.visibility != "external")
        // skip internal and private
        continue;
      const std::string func_name = method.name;
      if (func_name == c_name)
        // skip constructor
        continue;
      if (func_name == "receive" || func_name == "fallback")
        // skip receive and fallback
        continue;

      // then: function_call
      // do member access
      exprt mem_access = member_exprt(contract_var, method.id, method.type);

      // find function definition json node
      nlohmann::json decl_ref = get_func_decl_ref(c_name, func_name);
      if (decl_ref.empty())
      {
        log_error(
          "Internal error: fail to find the definition of function {}",
          method.name);
        abort();
      }

      side_effect_expr_function_callt then_expr;
      if (get_non_library_function_call(decl_ref, empty_json, then_expr))
        return true;

      // set &_ESBMC_tmp as the first argument
      then_expr.arguments().at(0) = contract_var;
      convert_expression_to_code(then_expr);

      code_blockt then;
      then.copy_to_operands(then_expr, return_expr);

      // ifthenelse-statement:
      codet if_expr("ifthenelse");
      if_expr.copy_to_operands(nondet_bool_expr, then);

      func_body.copy_to_operands(if_expr);
    }

    // 3. construct harness
    symbolt new_symbol;
    code_typet h_type;
    typet e_type = empty_typet();
    e_type.set("cpp_type", "void");
    h_type.return_type() = e_type;
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    get_default_symbol(
      new_symbol, debug_modulename, h_type, h_name, h_id, locationt());

    new_symbol.lvalue = true;
    new_symbol.is_extern = false;
    new_symbol.file_local = false;

    symbolt &added_sym = *context.move_symbol_to_context(new_symbol);

    // no params
    h_type.make_ellipsis();

    added_sym.type = h_type;
    added_sym.value = func_body;
    h_sym = added_sym;
  }

  sym = h_sym;
  return false;
}

// Normally, we would expect expr to be a code_declt expression
void solidity_convertert::move_to_initializer(const exprt &expr)
{
  // the initializer will clear its elements, so we populate the copy instead of origins
  if (!ctor_frontBlockDecl.operands().empty())
  {
    // reverse order
    for (auto &op : ctor_frontBlockDecl.operands())
    {
      convert_expression_to_code(op);
      initializers.copy_to_operands(op);
    }
    ctor_frontBlockDecl.clear();
  }

  initializers.copy_to_operands(expr);

  if (!ctor_backBlockDecl.operands().empty())
  {
    // reverse order
    for (auto &op : ctor_backBlockDecl.operands())
    {
      convert_expression_to_code(op);
      initializers.copy_to_operands(op);
    }
    ctor_backBlockDecl.clear();
  }
}

bool solidity_convertert::move_initializer_to_ctor(
  const nlohmann::json *based_contracts,
  const std::string contract_name)
{
  return move_initializer_to_ctor(based_contracts, contract_name, false);
}

// move library initializer and bind-name initializer to main
bool solidity_convertert::move_initializer_to_main(codet &func_body)
{
  side_effect_expr_function_callt call;
  get_library_function_call_no_args(
    "initialize", "c:@F@initialize", empty_typet(), locationt(), call);
  convert_expression_to_code(call);
  func_body.move_to_operands(call);

  for (auto str : contractNamesList)
  {
    std::string fname, fid;
    get_bind_cname_func_name(str, fname, fid);
    if (context.find_symbol(fid) == nullptr)
      return true;
    exprt func = symbol_expr(*context.find_symbol(fid));
    side_effect_expr_function_callt _call;
    _call.function() = func;
    convert_expression_to_code(_call);
    func_body.move_to_operands(_call);
  }
  return false;
}

// convert the initialization of the state variable
// into the equivalent assignmment in the ctor
// for inheritance_ctor, we skip the builtin assignment
bool solidity_convertert::move_initializer_to_ctor(
  const nlohmann::json *based_contracts,
  const std::string contract_name,
  bool is_aux_ctor)
{
  log_debug(
    "solidity",
    "@@@ Moving initialization of the state variable to the constructor {}().",
    contract_name);

  std::string ctor_id;
  if (is_aux_ctor)
  {
    exprt dump;
    get_inherit_ctor_definition(contract_name, dump);
    ctor_id = dump.identifier().as_string();
  }
  else
  {
    if (get_ctor_call_id(contract_name, ctor_id))
    {
      log_error("cannot find the construcor");
      return true;
    }
  }

  if (context.find_symbol(ctor_id) == nullptr)
  {
    log_error("cannot find the ctor ref of {}", ctor_id);
    return true;
  }
  symbolt &sym = *context.find_symbol(ctor_id);

  // get this pointer
  exprt base;
  if (get_func_decl_this_ref(contract_name, ctor_id, base))
  {
    log_error("cannot find function's this pointer");
    return true;
  }

  // queue insert initialization of the state
  for (auto it = initializers.operands().rbegin();
       it != initializers.operands().rend();
       ++it)
  {
    if (
      it->type().is_code() &&
      to_code(*it).get_statement().as_string() == "decl")
    {
      exprt comp = to_code_decl(to_code(*it)).op0();
      log_debug(
        "solidity",
        "\t@@@ initializing symbol {} in the constructor",
        comp.name().as_string());

      bool is_state = comp.type().get("#sol_state_var") == "1";
      if (!is_state)
      {
        // auxiliary local variable we created
        exprt tmp = *it;
        sym.value.operands().insert(sym.value.operands().begin(), tmp);
        continue;
      }
      if (is_aux_ctor)
      {
        if (
          comp.name().empty() ||
          is_sol_builin_symbol(contract_name, comp.name().as_string()))
          continue;
      }

      exprt lhs = member_exprt(base, comp.name(), comp.type());
      if (context.find_symbol(comp.identifier()) == nullptr)
      {
        log_error("Interal Error: cannot find symbol");
        abort();
      }
      symbolt *symbol = context.find_symbol(comp.identifier());
      exprt rhs = symbol->value;
      exprt _assign;
      if (
        lhs.type().get("#sol_type") == "STRING" &&
        rhs.get("#zero_initializer") != "1")
        get_string_assignment(lhs, rhs, _assign);
      else
      {
        _assign = side_effect_exprt("assign", comp.type());
        _assign.location() = sym.location;
        convert_type_expr(ns, rhs, comp.type());
        _assign.copy_to_operands(lhs, rhs);
      }
      convert_expression_to_code(_assign);

      // insert before the sym.value.operands
      sym.value.operands().insert(sym.value.operands().begin(), _assign);
    }
    else
    {
      exprt tmp = *it;
      convert_expression_to_code(tmp);
      sym.value.operands().insert(sym.value.operands().begin(), tmp);
    }
  }

  // insert parent ctor call in the front
  if (move_inheritance_to_ctor(based_contracts, contract_name, ctor_id, sym))
    return true;

  if (is_aux_ctor)
  {
    // hide it
    code_labelt label;
    label.set_label("__ESBMC_HIDE");
    label.code() = code_skipt();
    sym.value.operands().insert(sym.value.operands().begin(), label);
  }

  return false;
}

void solidity_convertert::move_to_front_block(const exprt &expr)
{
  if (current_functionDecl)
    expr_frontBlockDecl.copy_to_operands(expr);
  else
    ctor_frontBlockDecl.copy_to_operands(expr);
}

void solidity_convertert::move_to_back_block(const exprt &expr)
{
  if (current_functionDecl)
    expr_backBlockDecl.copy_to_operands(expr);
  else
    ctor_backBlockDecl.copy_to_operands(expr);
}

bool solidity_convertert::move_inheritance_to_ctor(
  const nlohmann::json *based_contracts,
  const std::string contract_name,
  std::string ctor_id,
  symbolt &sym)
{
  log_debug(
    "solidity",
    "@@@ Moving parents' constructor calls to the current constructor");

  std::string this_id = ctor_id + "#this";
  if (context.find_symbol(this_id) == nullptr)
  {
    log_error("Failed to find ctor this pointer {}", this_id);
    return true;
  }
  exprt this_expr = symbol_expr(*context.find_symbol(this_id));

  // queue insert the ctor initializaiton based on the linearizedBaseList
  if (based_contracts != nullptr && context.find_symbol(this_id) != nullptr)
  {
    /*
      Constructors are executed in the following order:
      1 - Base2
      2 - Base1
      3 - Derived3
      contract Derived3 is Base2, Base1 {
          constructor() Base1() Base2() {}
        }

      E.g. 
        contract DD is BB(3)
      Result ctor symbol table:
        Symbol......: c:@S@DD@F@DD#
        Module......: 1
        Base name...: DD
        Mode........: C++
        Type........: constructor  (struct DD *)
        Value.......: 
        {
          BB((struct BB *)this, 3);
        }
      However, since the c++ frontend is broken(esbmc/issues/1866),
      we convert it as 
        function ctor()
        {
          // create temporary object
          Base2 _ESBMC_ctor_Base2_tmp = new Base();
          // copy value
          this.x =  _ESBMC_ctor_Base2_tmp.x ;
          ...
        }
    */

    const std::vector<int> &id_list = linearizedBaseList[contract_name];
    for (auto it = id_list.begin() + 1; it != id_list.end(); ++it)
    {
      // handling inheritance
      // skip the first one as it is the contract itself
      std::string target_c_name = contractNamesMap[*it];

      for (const auto &c_node : (*based_contracts))
      {
        assert(c_node.contains("baseName"));
        std::string c_name = c_node["baseName"]["name"].get<std::string>();
        if (c_name != target_c_name)
          continue;

        typet c_type(irept::id_symbol);
        c_type.identifier(prefix + c_name);

        // get value
        // search for the parameter list for the constructor
        // they could be in two places:
        // - contract DD is BB(3)
        // or
        // - constructor() BB(3)
        nlohmann::json c_args_list_node = empty_json;
        const nlohmann::json &ctor_node = find_constructor_ref(contract_name);

        if (c_node.contains("arguments"))
          c_args_list_node = c_node;
        else if (!ctor_node.empty())
        {
          auto _ctor = ctor_node["modifiers"];
          for (const auto &c_mdf : _ctor)
          {
            if (
              !c_mdf.contains("modifierName") ||
              c_mdf["kind"] != "baseConstructorSpecifier")
              continue;

            if (c_mdf["modifierName"]["name"].get<std::string>() == c_name)
            {
              c_args_list_node = c_mdf;
              break;
            }
          }
        }

        // BB _ESBMC_aux_BB = BB(&this, 3, true);
        symbolt added_ctor_symbol;
        get_inherit_static_contract_instance(
          contract_name, c_name, c_args_list_node, added_ctor_symbol);

        // copy value e.g.  this.data = X.data
        struct_typet type_complete =
          to_struct_type(context.find_symbol(prefix + contract_name)->type);
        struct_typet c_type_complete =
          to_struct_type(context.find_symbol(prefix + c_name)->type);

        exprt lhs;
        exprt rhs;
        exprt _assign;
        for (const auto &c_comp : c_type_complete.components())
        {
          for (const auto &comp : type_complete.components())
          {
            if (c_comp.name() == comp.name())
            {
              assert(!comp.name().empty());
              assert(!c_comp.name().empty());

              if (is_sol_builin_symbol(c_name, c_comp.name().as_string()))
                // skip builtin symbol.
                //e.g. this->$address = _ESBMC_ctor_A_tmp.$address;
                continue;

              lhs = member_exprt(this_expr, comp.name(), comp.type());
              rhs = member_exprt(
                symbol_expr(added_ctor_symbol), c_comp.name(), c_comp.type());
              if (comp.type().get("#sol_type") == "STRING")
                get_string_assignment(lhs, rhs, _assign);
              else
              {
                _assign = side_effect_exprt("assign", comp.type());

                convert_type_expr(ns, rhs, comp.type());
                _assign.copy_to_operands(lhs, rhs);
              }

              convert_expression_to_code(_assign);
              sym.value.operands().insert(
                sym.value.operands().begin(), _assign);
              break;
            }
          }
        }

        // insert ctor call
        code_declt dl(symbol_expr(added_ctor_symbol));
        dl.operands().push_back(added_ctor_symbol.value);
        sym.value.operands().insert(sym.value.operands().begin(), dl);
      }
    }
  }
  return false;
}

bool solidity_convertert::get_access_from_decl(
  const nlohmann::json &ast_node,
  struct_typet::componentt &comp)
{
  if (
    SolidityGrammar::get_access_t(ast_node) ==
    SolidityGrammar::VisibilityT::UnknownT)
    return true;

  std::string access = ast_node["visibility"].get<std::string>();
  comp.set_access(access);

  return false;
}

bool solidity_convertert::get_function_definition(
  const nlohmann::json &ast_node)
{
  // For Solidity rule function-definition:
  // Order matters! do not change!
  // 1. Check fd.isImplicit() --- skipped since it's not applicable to Solidity
  // 2. Check fd.isDefined() and fd.isThisDeclarationADefinition()

  // Check intrinsic functions
  if (check_intrinsic_function(ast_node))
    return false;

  const nlohmann::json *old_functionDecl = current_functionDecl;
  const std::string old_functionName = current_functionName;

  current_functionDecl = &ast_node;

  bool is_ctor = (*current_functionDecl)["name"].get<std::string>() == "" &&
                 (*current_functionDecl).contains("kind") &&
                 (*current_functionDecl)["kind"] == "constructor";

  bool is_receive_fallback =
    (*current_functionDecl)["name"].get<std::string>() == "" &&
    (*current_functionDecl).contains("kind") &&
    ((*current_functionDecl)["kind"] == "receive" ||
     (*current_functionDecl)["kind"] == "fallback");

  std::string c_name;
  get_current_contract_name(ast_node, c_name);

  if (is_ctor)
    // for construcotr
    current_functionName = c_name;
  else if (is_receive_fallback)
    current_functionName = (*current_functionDecl)["kind"].get<std::string>();
  else
    current_functionName = (*current_functionDecl)["name"].get<std::string>();
  assert(!current_functionName.empty());

  // 4. Return type
  code_typet type;
  if (is_ctor)
  {
    typet tmp_rtn_type("constructor");
    type.return_type() = tmp_rtn_type;
    type.set("#member_name", prefix + c_name);
    type.set("#inlined", true);
  }
  else if (ast_node.contains("returnParameters"))
  {
    if (get_type_description(ast_node["returnParameters"], type.return_type()))
      return true;
    //? set member name?
  }
  else
  {
    type.return_type() = empty_typet();
    type.return_type().set("cpp_type", "void");
    type.set("#member_name", prefix + c_name);
  }

  // special handling for tuple:
  // construct a tuple type and a tuple instance
  if (type.return_type().get("#sol_type") == "TUPLE_RETURNS")
  {
    exprt dump;
    if (get_tuple_definition(*current_functionDecl))
      return true;
    if (get_tuple_instance(*current_functionDecl, dump))
      return true;
    type.return_type().set("#sol_tuple_id", dump.identifier().as_string());
  }

  // 5. Check fd.isVariadic(), fd.isInlined()
  //  Skipped since Solidity does not support variadic (optional args) or inline function.
  //  Actually "inline" doesn not make sense in Solidity

  // 6. Populate "locationt location_begin"
  locationt location_begin;
  get_location_from_node(ast_node, location_begin);

  // 7. Populate "std::string id, name"
  std::string name, id;
  get_function_definition_name(ast_node, name, id);
  assert(!name.empty());
  log_debug(
    "solidity",
    "\t@@@ Parsing function {} in contract {}",
    id.c_str(),
    current_baseContractName);

  if (context.find_symbol(id) != nullptr)
  {
    current_functionDecl = old_functionDecl;
    current_functionName = old_functionName;
    return false;
  }

  // 8. populate "std::string debug_modulename"
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  // 9. Populate "symbol.static_lifetime", "symbol.is_extern" and "symbol.file_local"
  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, type, name, id, location_begin);

  symbol.lvalue = true;
  symbol.is_extern =
    false; // TODO: hard coded for now, may need to change later
  symbol.file_local = true;

  // 10. Add symbol into the context
  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // 11. Convert parameters, if no parameter, assume ellipis
  //  - Convert params before body as they may get referred by the statement in the body

  // 11.1 add this pointer as the first param
  bool is_event_err_lib =
    ast_node.contains("nodeType") &&
    (ast_node["nodeType"] == "EventDefinition" ||
     ast_node["nodeType"] == "ErrorDefinition" ||
     SolidityGrammar::is_sol_library_function(ast_node["id"].get<int>()));
  if (!is_event_err_lib)
    get_function_this_pointer_param(
      c_name, id, debug_modulename, location_begin, type);

  // 11.2 parse other params
  SolidityGrammar::ParameterListT params =
    SolidityGrammar::get_parameter_list_t(ast_node["parameters"]);
  if (params != SolidityGrammar::ParameterListT::EMPTY)
  {
    // convert parameters if the function has them
    // update the typet, since typet contains parameter annotations
    for (const auto &decl : ast_node["parameters"]["parameters"].items())
    {
      log_debug(
        "solidity",
        "\t parsing function {}'s parameters",
        current_functionName);
      const nlohmann::json &func_param_decl = decl.value();

      code_typet::argumentt param;
      if (get_function_params(func_param_decl, c_name, param))
        return true;

      type.arguments().push_back(param);
    }
  }

  added_symbol.type = type;

  // 12. Convert body and embed the body into the same symbol
  // skip for 'unimplemented' functions which has no body,
  // e.g. asbstract/interface, the symbol value would be left as unset

  exprt body_exprt = code_blockt();
  if (
    ast_node.contains("body") ||
    (ast_node.contains("implemented") && ast_node["implemented"] == true))
  {
    log_debug(
      "solidity", "\t parsing function {}'s body", current_functionName);
    bool add_reentry = is_reentry_check && !is_event_err_lib && !is_ctor;

    if (has_modifier_invocation(ast_node))
    {
      // func() modf_1 modf_2
      // => func() => func_modf1() => func_modf2()
      if (get_func_modifier(
            ast_node, c_name, name, id, add_reentry, body_exprt))
        return true;
    }
    else
    {
      if (get_block(ast_node["body"], body_exprt))
        return true;
      if (add_reentry)
      {
        if (add_reentry_check(c_name, location_begin, body_exprt))
          return true;
      }
    }
  }

  added_symbol.value = body_exprt;

  //assert(!"done - finished all expr stmt in function?");

  // 13. Restore current_functionDecl
  log_debug("solidity", "Finish parsing function {}", current_functionName);
  current_functionDecl =
    old_functionDecl; // for __ESBMC_assume, old_functionDecl == null
  current_functionName = old_functionName;

  return false;
}

bool solidity_convertert::delete_modifier_json(
  const std::string &cname,
  const std::string &fname,
  nlohmann::json *&modifier_def)
{
  if (!src_ast_json.contains("nodes") || !src_ast_json["nodes"].is_array())
    return true;

  for (auto &node : src_ast_json["nodes"])
  {
    if (
      node.contains("name") && node["name"] == cname &&
      node.contains("nodeType") && node["nodeType"] == "ContractDefinition" &&
      node.contains("nodes") && node["nodes"].is_array())
    {
      nlohmann::json &contract_nodes = node["nodes"];

      for (auto it = contract_nodes.begin(); it != contract_nodes.end(); ++it)
      {
        if (
          it->contains("nodeType") &&
          (*it)["nodeType"] == "FunctionDefinition" && it->contains("name") &&
          (*it)["name"] == fname)
        {
          contract_nodes.erase(it);
          modifier_def = nullptr;
          return false;
        }
      }
    }
  }
  return true;
}

bool solidity_convertert::insert_modifier_json(
  const nlohmann::json &ast_node,
  const std::string &cname,
  const std::string &fname,
  nlohmann::json *&modifier_def)
{
  log_debug("solidity", "\tinsert modifier json {}", fname);
  nlohmann::json &nodes = src_ast_json["nodes"];
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    if (
      (*itr).contains("name") && (*itr)["name"].get<std::string>() == cname &&
      (*itr)["nodeType"] == "ContractDefinition")
    {
      nlohmann::json &contract_nodes = (*itr)["nodes"];

      // check if we already inserted
      for (auto &func_node : contract_nodes)
      {
        if (
          func_node.contains("nodeType") &&
          func_node["nodeType"] == "FunctionDefinition" &&
          func_node.contains("name") && func_node["name"] == fname)
        {
          modifier_def = &func_node;
          return false;
        }
      }

      const nlohmann::json returnParameters = ast_node["returnParameters"];
      const nlohmann::json src = ast_node["src"];
      /*
        append below node to the contract_nodes
        {
          nodeType: FunctionDefinition
          id: 0
          name: fname
          returnParameters: returnParameters
          src: src
        }
      */
      nlohmann::json new_function = {
        {"nodeType", "FunctionDefinition"},
        {"id", 0},
        {"name", fname},
        {"kind", "function"},
        {"implemented", true},
        {"returnParameters", returnParameters},
        {"src", src}};

      contract_nodes.push_back(new_function);
      modifier_def = &contract_nodes.back();
      return false;
    }
  }

  return true; // unexpected error
}

void solidity_convertert::get_modifier_function_name(
  const std::string &cname,
  const std::string &mod_name,
  const std::string &func_name,
  std::string &name,
  std::string &id)
{
  name = func_name + "_" + mod_name;
  id = "sol:@C@" + cname + "@F@" + name + "#0";
}

bool solidity_convertert::has_modifier_invocation(
  const nlohmann::json &ast_node)
{
  // check if there is any modifier invocation (not constructor calls)
  if (!ast_node.contains("modifiers") || ast_node["modifiers"].empty())
  {
    return false;
  }

  nlohmann::json modifiers = nlohmann::json::array();
  for (const auto &m : ast_node["modifiers"])
  {
    assert(m.contains("kind"));
    if (m.contains("kind") && m["kind"] == "modifierInvocation")
    {
      modifiers.push_back(m);
    }
  }
  // if there is no modifier invocation, return
  if (modifiers.size() == 0)
    return false;

  return true;
}

bool solidity_convertert::add_reentry_check(
  const std::string &c_name,
  const locationt &loc,
  exprt &body_exprt)
{
  // we should only add this to the contract's functions
  // rather than interface and library's functions,
  // or contract's errors, events and ctor
  //TODO: detect is_library_function

  // add a global mutex checker _ESBMC_check_reentrancy() in the front
  side_effect_expr_function_callt call;
  get_library_function_call_no_args(
    "_ESBMC_check_reentrancy",
    "c:@F@_ESBMC_check_reentrancy",
    empty_typet(),
    loc,
    call);

  exprt this_expr;
  if (get_func_decl_this_ref(*current_functionDecl, this_expr))
    return true;

  exprt arg;
  get_contract_mutex_expr(c_name, this_expr, arg);
  call.arguments().push_back(arg);

  convert_expression_to_code(call);
  // Insert after the last front requirement (__ESBMC_assume) statement,
  // as the function may only be re-entered once the requirements are fulfilled.
  auto &ops = body_exprt.operands();
  for (auto it = ops.begin(); it != ops.end(); ++it)
  {
    if (
      it->op0().id() == "sideeffect" &&
      it->op0().op0().name() == "__ESBMC_assume")
      continue;

    ops.insert(it, call);
    break;
  }
  return false;
}

// parse the modifiers, this could be:
// 1. merge code into function body
// 2. construct an auxiliary function, move the body to it, and call it
bool solidity_convertert::get_func_modifier(
  const nlohmann::json &ast_node,
  const std::string &c_name,
  const std::string &f_name,
  const std::string &f_id,
  const bool add_reentry,
  exprt &body_exprt)
{
  log_debug("solidity", "parsing function modifiers");
  nlohmann::json modifiers = nlohmann::json::array();
  for (const auto &m : ast_node["modifiers"])
  {
    if (m.contains("kind") && m["kind"] == "modifierInvocation")
      modifiers.push_back(m);
  }
  // rebegin the function body
  for (auto it = modifiers.rbegin(); it != modifiers.rend(); ++it)
  {
    int modifier_id = (*it)["modifierName"]["referencedDeclaration"];
    // we cannot use reference here, as the src_ast_json got inserted/deleted later
    const nlohmann::json mod_def =
      find_decl_ref(src_ast_json["nodes"], modifier_id);
    assert(!mod_def.is_null());
    assert(!mod_def.empty());

    std::string func_name = f_name;
    std::string mod_name = mod_def["name"];
    std::string aux_func_name, aux_func_id;
    get_modifier_function_name(
      c_name, mod_name, func_name, aux_func_name, aux_func_id);

    nlohmann::json *modifier_func = nullptr;
    if (insert_modifier_json(ast_node, c_name, aux_func_name, modifier_func))
      return true;
    assert(modifier_func != nullptr);
    auto old_decl = current_functionDecl;
    auto old_name = current_functionName;
    current_functionDecl = modifier_func;
    current_functionName = aux_func_name;

    symbolt added_sym;
    locationt loc;
    get_location_from_node(ast_node, loc);
    std::string debug_mode = get_modulename_from_path(absolute_path);
    code_typet aux_type;

    bool has_return = ast_node.contains("returnParameters") &&
                      !ast_node["returnParameters"]["parameters"].empty();

    if (has_return)
    {
      // return func_modifier();
      if (get_type_description(
            ast_node["returnParameters"], aux_type.return_type()))
        return true;
    }
    else
    {
      aux_type.return_type() = empty_typet();
      aux_type.set("cpp_type", "void");
    }

    get_default_symbol(
      added_sym, debug_mode, aux_type, aux_func_name, aux_func_id, loc);
    added_sym.lvalue = true;
    added_sym.file_local = true;

    // move the symbol to the context
    symbolt &a_sym = *move_symbol_to_context(added_sym);
    get_function_this_pointer_param(
      c_name, aux_func_id, debug_mode, loc, aux_type);
    for (const auto &param : mod_def["parameters"]["parameters"])
    {
      code_typet::argumentt arg;
      if (get_function_params(param, c_name, arg))
        return true;
      aux_type.arguments().push_back(arg);
    }
    a_sym.type = aux_type;
    move_builtin_to_contract(c_name, symbol_expr(a_sym), "internal", true);

    // same as origin function body
    if (body_exprt.operands().empty())
    {
      // modify the src_ast_json: insert the func node
      // nodeType: esbmcModfunction
      if (get_block(ast_node["body"], body_exprt))
        return true;
      if (add_reentry)
      {
        if (add_reentry_check(c_name, loc, body_exprt))
          return true;
      }
    }
    else
    {
      // this can only be a function call
      assert(body_exprt.operands().size() == 1);
    }

    // get func body
    code_blockt mod_body;
    if (get_block(mod_def["body"], mod_body))
      return true;

    for (auto &stmt : mod_body.operands())
    {
      if (stmt.get_bool("#is_modifier_placeholder"))
        stmt = body_exprt;
    }
    a_sym.value = mod_body;

    // reset
    current_functionDecl = old_decl;
    current_functionName = old_name;
    if (delete_modifier_json(c_name, aux_func_name, modifier_func))
      return true;
    assert(modifier_func == nullptr);

    // construct the function call
    side_effect_expr_function_callt func_modifier;
    func_modifier.function() = symbol_expr(a_sym);

    exprt this_ptr;
    auto next_it = std::next(it);
    if (next_it != modifiers.rend())
    {
      int next_modifier_id =
        (*next_it)["modifierName"]["referencedDeclaration"];
      const nlohmann::json &next_mod_def =
        find_decl_ref(src_ast_json["nodes"], next_modifier_id);

      std::string next_mod_name = next_mod_def["name"];
      std::string next_aux_func_name, next_aux_func_id;
      get_modifier_function_name(
        c_name, next_mod_name, f_name, next_aux_func_name, next_aux_func_id);

      if (get_func_decl_this_ref(c_name, next_aux_func_id, this_ptr))
        return true;
      func_modifier.arguments().push_back(this_ptr);

      if (insert_modifier_json(
            ast_node, c_name, next_aux_func_name, modifier_func))
        return true;
      assert(modifier_func != nullptr);
      auto old_decl = current_functionDecl;
      auto old_name = current_functionName;
      current_functionDecl = modifier_func;
      current_functionName = next_aux_func_name;

      for (const auto &arg_json : (*it)["arguments"])
      {
        exprt arg_expr;
        if (get_expr(arg_json, arg_json["typeDescriptions"], arg_expr))
          return true;
        func_modifier.arguments().push_back(arg_expr);
      }

      // reset
      current_functionDecl = old_decl;
      current_functionName = old_name;
    }
    else
    {
      // original
      if (get_func_decl_this_ref(c_name, f_id, this_ptr))
        return true;
      func_modifier.arguments().push_back(this_ptr);

      for (const auto &arg_json : (*it)["arguments"])
      {
        exprt arg_expr;
        if (get_expr(arg_json, arg_json["typeDescriptions"], arg_expr))
          return true;
        func_modifier.arguments().push_back(arg_expr);
      }
    }

    code_blockt _block;
    if (has_return)
    {
      // return func_modifier();
      code_returnt return_expr = code_returnt();
      return_expr.return_value() = func_modifier;
      _block.move_to_operands(return_expr);
      body_exprt = _block;
    }
    else
    {
      convert_expression_to_code(func_modifier);
      _block.move_to_operands(func_modifier);
      body_exprt = _block;
    }
  }

  return false;
}

void solidity_convertert::reset_auxiliary_vars()
{
  current_baseContractName = "";
  current_functionName = "";
  current_functionDecl = nullptr;
  current_forStmt = nullptr;
  current_typeName = nullptr;
  initializers.clear();
}

bool solidity_convertert::get_function_params(
  const nlohmann::json &pd,
  const std::string &cname,
  exprt &param)
{
  // 1. get parameter type
  typet param_type;
  if (get_type_description(pd["typeDescriptions"], param_type))
    return true;

  // 2a. get id and name
  std::string id, name;
  assert(current_functionName != ""); // we are converting a function param now
  assert(current_functionDecl);
  get_local_var_decl_name(pd, cname, name, id);
  // 2b. handle Omitted Names in Function Definitions
  if (name == "")
  {
    // Items with omitted names will still be present on the stack, but they are inaccessible by name.
    // e.g. ~omitted1, ~omitted2. which is a invalid name for solidity.
    // Therefore it won't conflict with other arg names.
    //log_error("Omitted params are not supported");
    // return true;
    ;
  }

  param = code_typet::argumentt();
  param.type() = param_type;
  param.cmt_base_name(name);

  // 3. get location
  locationt location_begin;
  get_location_from_node(pd, location_begin);

  param.cmt_identifier(id);
  param.location() = location_begin;

  // 4. get symbol
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());
  symbolt param_symbol;
  get_default_symbol(
    param_symbol, debug_modulename, param_type, name, id, location_begin);

  // 5. set symbol's lvalue, is_parameter and file local
  param_symbol.lvalue = true;
  param_symbol.is_parameter = true;
  param_symbol.file_local = true;

  // 6. add symbol to the context
  move_symbol_to_context(param_symbol);

  return false;
}

bool solidity_convertert::get_block(
  const nlohmann::json &block,
  exprt &new_expr)
{
  // For rule block
  locationt location;
  get_start_location_from_stmt(block, location);

  SolidityGrammar::BlockT type = SolidityGrammar::get_block_t(block);
  log_debug(
    "solidity",
    "	@@@ got Block: SolidityGrammar::BlockT::{}",
    SolidityGrammar::block_to_str(type));

  switch (type)
  {
  // equivalent to clang::Stmt::CompoundStmtClass
  // deal with a block of statements
  case SolidityGrammar::BlockT::Statement:
  {
    const nlohmann::json &stmts = block["statements"];

    code_blockt _block;
    unsigned ctr = 0;
    // items() returns a key-value pair with key being the index
    for (auto const &stmt_kv : stmts.items())
    {
      locationt cl;
      get_location_from_node(stmt_kv.value(), cl);

      exprt statement;
      if (get_statement(stmt_kv.value(), statement))
        return true;

      if (!expr_frontBlockDecl.operands().empty())
      {
        for (auto op : expr_frontBlockDecl.operands())
        {
          convert_expression_to_code(op);
          _block.operands().push_back(op);
        }
        expr_frontBlockDecl.clear();
      }
      statement.location() = cl;
      convert_expression_to_code(statement);
      _block.operands().push_back(statement);

      if (!expr_backBlockDecl.operands().empty())
      {
        for (auto op : expr_backBlockDecl.operands())
        {
          convert_expression_to_code(op);
          _block.operands().push_back(op);
        }
        expr_backBlockDecl.clear();
      }

      ++ctr;
    }
    log_debug("solidity", " \t@@@ CompoundStmt has {} statements", ctr);

    locationt location_end;
    get_final_location_from_stmt(block, location_end);

    _block.end_location(location_end);
    new_expr = _block;
    break;
  }
  case SolidityGrammar::BlockT::BlockForStatement:
  case SolidityGrammar::BlockT::BlockIfStatement:
  case SolidityGrammar::BlockT::BlockWhileStatement:
  {
    // this means only one statement in the block
    exprt statement;

    // pass directly to get_statement()
    if (get_statement(block, statement))
      return true;
    convert_expression_to_code(statement);
    new_expr = statement;
    break;
  }
  case SolidityGrammar::BlockT::BlockExpressionStatement:
  {
    get_expr(block["expression"], new_expr);
    break;
  }
  case SolidityGrammar::BlockT::BlockTError:
  default:
  {
    assert(!"Unimplemented type in rule block");
    return true;
  }
  }

  new_expr.location() = location;
  return false;
}

bool solidity_convertert::get_statement(
  const nlohmann::json &stmt,
  exprt &new_expr)
{
  // For rule statement
  // Since this is an additional layer of grammar rules compared to clang C, we do NOT set location here.
  // Just pass the new_expr reference to the next layer.

  locationt loc;
  get_location_from_node(stmt, loc);

  SolidityGrammar::StatementT type = SolidityGrammar::get_statement_t(stmt);
  log_debug(
    "solidity",
    "	@@@ got Stmt: SolidityGrammar::StatementT::{}",
    SolidityGrammar::statement_to_str(type));

  switch (type)
  {
  case SolidityGrammar::StatementT::Block:
  {
    if (get_block(stmt, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::StatementT::ExpressionStatement:
  {
    if (get_expr(
          stmt["expression"], stmt["expression"]["typeDescriptions"], new_expr))
      return true;
    break;
  }
  case SolidityGrammar::StatementT::VariableDeclStatement:
  {
    const nlohmann::json &declgroup = stmt["declarations"];

    codet decls("decl-block");
    unsigned ctr = 0;
    // N.B. Although Solidity AST JSON uses "declarations": [],
    // the size of this array is alway 1!
    // A second declaration will become another stmt in "statements" array
    // e.g. "statements" : [
    //  {"declarations": [], "id": 1}
    //  {"declarations": [], "id": 2}
    //  {"declarations": [], "id": 3}
    // ]
    if (declgroup.size() == 1)
    {
      // deal with local var decl with init value
      const nlohmann::json &decl = declgroup[0];
      nlohmann::json initialValue = nlohmann::json::object();
      if (stmt.contains("initialValue"))
        initialValue = stmt["initialValue"];

      exprt single_decl;
      if (get_var_decl(decl, initialValue, single_decl))
        return true;

      decls.operands().push_back(single_decl);
      ++ctr;
    }
    else
    {
      // seperate the decl and assignment
      for (const auto &it : declgroup.items())
      {
        if (it.value().is_null() || it.value().empty())
          continue;
        const nlohmann::json &decl = it.value();
        exprt single_decl;
        if (get_var_decl(decl, single_decl))
          return true;
        decls.operands().push_back(single_decl);
        ++ctr;
      }

      if (stmt.contains("initialValue"))
      {
        // this is a tuple expression
        const nlohmann::json &initialValue = stmt["initialValue"];
        exprt tuple_expr;
        if (get_expr(initialValue, tuple_expr))
          return true;

        code_blockt lhs_block;
        for (auto &decl : decls.operands())
          lhs_block.copy_to_operands(decl.op0()); // nil_expr;

        //TODO FIXME: stmt might not contains right handside
        construct_tuple_assigments(stmt, lhs_block, tuple_expr);
      }
    }
    log_debug("solidity", " \t@@@ DeclStmt group has {} decls", ctr);

    new_expr = decls;
    break;
  }
  case SolidityGrammar::StatementT::ReturnStatement:
  {
    if (!current_functionDecl)
    {
      log_error(
        "Error: ESBMC could not find the parent scope for this "
        "ReturnStatement");
      return true;
    }

    // 1. get return type
    // TODO: Fix me! Assumptions:
    //  a). It's "return <expr>;" not "return;"
    //  b). <expr> is pointing to a DeclRefExpr, we need to wrap it in an ImplicitCastExpr as a subexpr
    //  c). For multiple return type, the return statement represented as a tuple expression using a components field.
    //      Besides, tuple can only be declared literally. https://docs.soliditylang.org/en/latest/control-structures.html#assignment
    //      e.g. return (false, 123)
    if (!stmt.contains("expression"))
    {
      // "return;"
      code_returnt ret_expr;
      new_expr = ret_expr;
      return false;
    }
    assert(stmt["expression"].contains("nodeType"));

    // get_type_description
    typet return_exrp_type;
    if (get_type_description(
          stmt["expression"]["typeDescriptions"], return_exrp_type))
      return true;

    if (return_exrp_type.get("#sol_type") == "TUPLE_RETURNS")
    {
      if (
        stmt["expression"]["nodeType"].get<std::string>() !=
          "TupleExpression" &&
        stmt["expression"]["nodeType"].get<std::string>() != "FunctionCall")
      {
        log_error("Unexpected tuple");
        return true;
      }

      // get tuple instance
      std::string tname, tid;
      if (get_tuple_instance_name(*current_functionDecl, tname, tid))
        return true;
      if (context.find_symbol(tid) == nullptr)
        return true;

      // get lhs
      exprt lhs = symbol_expr(*context.find_symbol(tid));

      if (
        stmt["expression"]["nodeType"].get<std::string>() == "TupleExpression")
      {
        // return (x,y) ==>
        // tuple.mem0 = x; tuple.mem1 = y; return ;

        // get rhs
        // hack: we need the expression block, not tuple instance
        current_lhsDecl = true;
        exprt rhs;
        if (get_expr(stmt["expression"], rhs))
          return true;
        current_lhsDecl = false;

        size_t ls = to_struct_type(lhs.type()).components().size();
        size_t rs = rhs.operands().size();
        if (ls != rs)
        {
          log_debug(
            "soldiity",
            "Handling return tuple.\nlhs = {}\nrhs = {}",
            lhs.to_string(),
            rhs.to_string());
          log_error("Internal tuple error.");
        }

        for (size_t i = 0; i < ls; i++)
        {
          // lop: struct member call (e.g. tuple.men0)
          exprt lop;
          if (get_tuple_member_call(
                lhs.identifier(),
                to_struct_type(lhs.type()).components().at(i),
                lop))
            return true;

          // rop: constant/symbol
          exprt rop = rhs.operands().at(i);

          // do assignment
          get_tuple_assignment(lop, rop);
        }
      }
      else
      {
        // return func(); ==>
        // tuple1.mem0 = tuple0.mem0; return;

        // get rhs
        exprt rhs;
        if (get_tuple_function_ref(stmt["expression"]["expression"], rhs))
          return true;

        // add function call
        exprt func_call;
        if (get_expr(
              stmt["expression"],
              stmt["expression"]["typeDescriptions"],
              func_call))
          return true;
        get_tuple_function_call(func_call);

        size_t ls = to_struct_type(lhs.type()).components().size();
        size_t rs = to_struct_type(rhs.type()).components().size();
        if (ls != rs)
        {
          log_error("Unexpected tuple structure");
          abort();
        }

        for (size_t i = 0; i < ls; i++)
        {
          // lop: struct member call (e.g. tupleA.men0)
          exprt lop;
          if (get_tuple_member_call(
                lhs.identifier(),
                to_struct_type(lhs.type()).components().at(i),
                lop))
            return true;

          // rop: struct member call (e.g. tupleB.men0)
          exprt rop;
          if (get_tuple_member_call(
                rhs.identifier(),
                to_struct_type(rhs.type()).components().at(i),
                rop))
            return true;

          // do assignment
          get_tuple_assignment(lop, rop);
        }
      }
      // do return in the end
      exprt return_expr = code_returnt();
      move_to_back_block(return_expr);

      new_expr = code_skipt();
      break;
    }

    typet return_type;
    if ((*current_functionDecl).contains("returnParameters"))
    {
      assert(
        (*current_functionDecl)["returnParameters"]["id"]
          .get<std::uint16_t>() ==
        stmt["functionReturnParameters"].get<std::uint16_t>());
      if (get_type_description(
            (*current_functionDecl)["returnParameters"], return_type))
        return true;
    }
    else
      return true;

    nlohmann::json literal_type = nullptr;

    auto expr_type = SolidityGrammar::get_expression_t(stmt["expression"]);
    bool expr_is_literal = expr_type == SolidityGrammar::Literal;
    if (expr_is_literal)
      literal_type = make_return_type_from_typet(return_type);

    // 2. get return value
    code_returnt ret_expr;
    const nlohmann::json &rtn_expr = stmt["expression"];
    // wrap it in an ImplicitCastExpr to convert LValue to RValue
    nlohmann::json implicit_cast_expr =
      make_implicit_cast_expr(rtn_expr, "LValueToRValue");

    /* There could be case like
      {
      "expression": {
          "kind": "number",
          "nodeType": "Literal",
          "typeDescriptions": {
              "typeIdentifier": "t_rational_11_by_1",
              "typeString": "int_const 11"
          },
          "value": "12345"
      },
      "nodeType": "Return",
      }
      Therefore, we need to pass the literal_type value.
      */

    exprt rhs;
    if (get_expr(implicit_cast_expr, literal_type, rhs))
      return true;

    solidity_gen_typecast(ns, rhs, return_type);
    ret_expr.return_value() = rhs;

    new_expr = ret_expr;

    break;
  }
  case SolidityGrammar::StatementT::ForStatement:
  {
    // Based on rule for-statement

    // For nested loop
    const nlohmann::json *old_forStmt = current_forStmt;
    current_forStmt = &stmt;

    // 1. annotate init
    codet init =
      code_skipt(); // code_skipt() means no init in for-stmt, e.g. for (; i< 10; ++i)
    if (stmt.contains("initializationExpression"))
      if (get_statement(stmt["initializationExpression"], init))
        return true;

    convert_expression_to_code(init);

    // 2. annotate condition
    exprt cond = true_exprt();
    if (stmt.contains("condition"))
      if (get_expr(stmt["condition"], cond))
        return true;

    // 3. annotate increment
    codet inc = code_skipt();
    if (stmt.contains("loopExpression"))
      if (get_statement(stmt["loopExpression"], inc))
        return true;

    convert_expression_to_code(inc);

    // 4. annotate body
    codet body = code_skipt();
    if (stmt.contains("body"))
      if (get_statement(stmt["body"], body))
        return true;

    convert_expression_to_code(body);

    code_fort code_for;
    code_for.init() = init;
    code_for.cond() = cond;
    code_for.iter() = inc;
    code_for.body() = body;

    new_expr = code_for;
    current_forStmt = old_forStmt;
    break;
  }
  case SolidityGrammar::StatementT::IfStatement:
  {
    // Based on rule if-statement
    // 1. Condition: make a exprt for condition
    exprt cond;
    if (get_expr(stmt["condition"], cond))
      return true;

    // 2. Then: make a exprt for trueBody
    exprt then;
    if (get_statement(stmt["trueBody"], then))
      return true;

    convert_expression_to_code(then);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(cond, then);

    // 3. Else: make a exprt for "falseBody" if the if-statement node contains an "else" block
    if (stmt.contains("falseBody"))
    {
      exprt else_expr;
      if (get_statement(stmt["falseBody"], else_expr))
        return true;

      convert_expression_to_code(else_expr);
      if_expr.copy_to_operands(else_expr);
    }

    new_expr = if_expr;
    break;
  }
  case SolidityGrammar::StatementT::WhileStatement:
  {
    exprt cond = true_exprt();
    if (get_expr(stmt["condition"], cond))
      return true;

    codet body = codet();
    if (get_block(stmt["body"], body))
      return true;

    convert_expression_to_code(body);

    code_whilet code_while;
    code_while.cond() = cond;
    code_while.body() = body;

    new_expr = code_while;
    break;
  }
  case SolidityGrammar::StatementT::ContinueStatement:
  {
    new_expr = code_continuet();
    break;
  }
  case SolidityGrammar::StatementT::BreakStatement:
  {
    new_expr = code_breakt();
    break;
  }
  case SolidityGrammar::StatementT::RevertStatement:
  {
    // e.g.
    // {
    //   "errorCall": {
    //     "nodeType": "FunctionCall",
    //   }
    //   "nodeType": "RevertStatement",
    // }
    if (!stmt.contains("errorCall") || get_expr(stmt["errorCall"], new_expr))
      return true;

    break;
  }
  case SolidityGrammar::StatementT::EmitStatement:
  {
    // treat emit as function call
    if (!stmt.contains("eventCall"))
    {
      log_error("Unexpected emit statement.");
      return true;
    }
    if (get_expr(stmt["eventCall"], new_expr))
      return true;

    break;
  }
  case SolidityGrammar::StatementT::PlaceholderStatement:
  {
    code_skipt placeholder;
    placeholder.set("#is_modifier_placeholder", true);
    new_expr = placeholder;
    break;
  }
  case SolidityGrammar::StatementT::StatementTError:
  default:
  {
    log_error(
      "Unimplemented Statement type in rule statement. Got {}",
      SolidityGrammar::statement_to_str(type));
    return true;
  }
  }

  log_debug(
    "solidity", "finish statement {}", SolidityGrammar::statement_to_str(type));
  new_expr.location() = loc;
  return false;
}

/**
     * @brief Populate the out parameter with the expression based on
     * the solidity expression grammar
     *
     * @param expr The expression ast is to be converted to the IR
     * @param new_expr Out parameter to hold the conversion
     * @return true iff the conversion has failed
     * @return false iff the conversion was successful
     */
bool solidity_convertert::get_expr(const nlohmann::json &expr, exprt &new_expr)
{
  return get_expr(expr, nullptr, new_expr);
}

/**
     * @brief Populate the out parameter with the expression based on
     * the solidity expression grammar. 
     * 
     * More specifically, parse each expression in the AST json and
     * convert it to a exprt ("new_expr"). The expression may have sub-expression
     * 
     * !Always check if the expression is a Literal before calling get_expr
     * !Unless you are 100% sure it will not be a constant
     * 
     * This function is called throught two paths:
     * 1. get_non_function_decl => get_var_decl => get_expr
     * 2. get_non_function_decl => get_function_definition => get_statement => get_expr
     * 
     * @param expr The expression that is to be converted to the IR
     * @param literal_type Type information ast to create the the literal
     * type in the IR (only needed for when the expression is a literal).
     * A literal_type is a "typeDescriptions" ast_node.
     * we need this due to some info is missing in the child node.
     * @param new_expr Out parameter to hold the conversion
     * @return true iff the conversion has failed
     * @return false iff the conversion was successful
     */
bool solidity_convertert::get_expr(
  const nlohmann::json &expr,
  const nlohmann::json &literal_type,
  exprt &new_expr)
{
  assert(literal_type.is_null() || !literal_type.contains("typeDescriptions"));
  // For rule expression
  // We need to do location settings to match clang C's number of times to set the locations when recurring
  locationt location;
  get_start_location_from_stmt(expr, location);

  std::string current_contractName;
  get_current_contract_name(expr, current_contractName);

  SolidityGrammar::ExpressionT type = SolidityGrammar::get_expression_t(expr);
  log_debug(
    "solidity",
    "  @@@ got Expr: SolidityGrammar::ExpressionT::{}",
    SolidityGrammar::expression_to_str(type));

  switch (type)
  {
  case SolidityGrammar::ExpressionT::BinaryOperatorClass:
  {
    if (get_binary_operator_expr(expr, new_expr))
      return true;

    break;
  }
  case SolidityGrammar::ExpressionT::UnaryOperatorClass:
  {
    if (get_unary_operator_expr(expr, literal_type, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::ConditionalOperatorClass:
  {
    // for Ternary Operator (...?...:...) only
    if (get_conditional_operator_expr(expr, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::DeclRefExprClass:
  {
    if (expr["referencedDeclaration"] > 0)
    {
      // Soldity uses +ve odd numbers to refer to var or functions declared in the contract
      const nlohmann::json &decl =
        find_decl_ref(src_ast_json, expr["referencedDeclaration"]);
      if (decl.empty())
      {
        log_error(
          "failed to find the reference AST node, base contract name {}, "
          "reference id {}",
          current_baseContractName,
          std::to_string(expr["referencedDeclaration"].get<int>()));
        return true;
      }

      if (!check_intrinsic_function(decl))
      {
        log_debug(
          "solidity",
          "\t\t@@@ got nodeType={}",
          decl["nodeType"].get<std::string>());
        if (decl["nodeType"] == "VariableDeclaration")
        {
          if (get_var_decl_ref(decl, true, new_expr))
            return true;
        }
        else if (decl["nodeType"] == "FunctionDefinition")
        {
          if (get_func_decl_ref(decl, new_expr))
            return true;
        }
        else if (
          decl["nodeType"] == "StructDefinition" ||
          decl["nodeType"] == "ErrorDefinition" ||
          decl["nodeType"] == "EventDefinition" ||
          (decl["nodeType"] == "ContractDefinition" &&
           decl["contractKind"] == "library"))
        {
          if (get_noncontract_decl_ref(decl, new_expr))
            return true;
        }
        else
        {
          log_error(
            "Unsupported DeclRefExprClass type, got nodeType={}",
            decl["nodeType"].get<std::string>());
          return true;
        }
      }
      else
      {
        // for special functions, we need to deal with it separately
        if (get_esbmc_builtin_ref(expr, new_expr))
          return true;
      }
    }
    else
    {
      if (expr.contains("name") && expr["name"] == "this")
      {
        log_debug("solidity", "\t\tgot this ref");

        exprt this_expr;
        assert(current_functionDecl);
        if (get_func_decl_this_ref(*current_functionDecl, this_expr))
          return true;
        new_expr = this_expr;
      }
      else
      {
        // Soldity uses -ve odd numbers to refer to built-in var or functions that
        // are NOT declared in the contract
        if (get_esbmc_builtin_ref(expr, new_expr))
          return true;
      }
    }

    break;
  }
  case SolidityGrammar::ExpressionT::LiteralWithRational:
  {
    // extract integer literal
    std::string typeString = expr["typeDescriptions"]["typeString"];
    // Remove "int_const " prefix
    std::string value_str = typeString.substr(10);

    BigInt z_ext_value = string2integer(value_str);
    unsignedbv_typet type(256);
    new_expr = constant_exprt(
      integer2binary(z_ext_value, bv_width(type)),
      integer2string(z_ext_value),
      type);

    break;
  }
  case SolidityGrammar::ExpressionT::Literal:
  {
    // make a type-name json for integer literal conversion
    std::string the_value = expr["value"].get<std::string>();
    const nlohmann::json &literal = expr["typeDescriptions"];
    SolidityGrammar::ElementaryTypeNameT type_name =
      SolidityGrammar::get_elementary_type_name_t(literal);
    log_debug(
      "solidity",
      "	@@@ got Literal: SolidityGrammar::ElementaryTypeNameT::{}",
      SolidityGrammar::elementary_type_name_to_str(type_name));

    if (
      literal_type != nullptr && literal_type.contains("typeString") &&
      literal_type["typeString"].get<std::string>().find("bytes") !=
        std::string::npos)
    {
      // literal_type["typeString"] could be
      //    "bytes1" ... "bytes32"
      //    "bytes storage ref"
      // e.g.
      //    bytes1 x = 0x12;
      //    bytes32 x = "string";
      //    bytes x = "string";
      //
      SolidityGrammar::ElementaryTypeNameT type =
        SolidityGrammar::get_elementary_type_name_t(literal_type);

      int byte_size;
      if (type == SolidityGrammar::ElementaryTypeNameT::BYTES)
        // dynamic bytes array, the type is set to unsignedbv(256);
        byte_size = 32;
      else
        byte_size = bytesn_type_name_to_size(type);

      // convert hex to decimal value and populate
      switch (type_name)
      {
      case SolidityGrammar::ElementaryTypeNameT::INT_LITERAL:
      {
        if (convert_hex_literal(the_value, new_expr, byte_size * 8))
          return true;

        new_expr.type().set("#sol_type", "BYTES_LITERAL");
        break;
      }
      case SolidityGrammar::ElementaryTypeNameT::STRING_LITERAL:
      {
        std::string hex_val = expr["hexValue"].get<std::string>();

        // add padding
        for (int i = 0; i < byte_size; i++)
          hex_val += "00";
        hex_val.resize(byte_size * 2);

        if (convert_hex_literal(hex_val, new_expr, byte_size * 8))
          return true;

        new_expr.type().set("#sol_type", "BYTES_LITERAL");
        break;
      }
      default:
        assert(!"Error occurred when handling bytes literal");
      }
      break;
    }

    switch (type_name)
    {
    case SolidityGrammar::ElementaryTypeNameT::INT_LITERAL:
    {
      assert(literal_type != nullptr);
      if (
        the_value.length() >= 2 &&
        the_value.substr(0, 2) == "0x") // meaning hex-string
      {
        if (convert_hex_literal(the_value, new_expr))
          return true;
        new_expr.type().set("#sol_type", "INT_CONST");
      }
      else if (convert_integer_literal(literal_type, the_value, new_expr))
        return true;

      break;
    }
    case SolidityGrammar::ElementaryTypeNameT::BOOL:
    {
      if (convert_bool_literal(literal, the_value, new_expr))
        return true;
      break;
    }
    case SolidityGrammar::ElementaryTypeNameT::STRING_LITERAL:
    {
      if (convert_string_literal(the_value, new_expr))
        return true;

      break;
    }
    case SolidityGrammar::ElementaryTypeNameT::ADDRESS:
    case SolidityGrammar::ElementaryTypeNameT::ADDRESS_PAYABLE:
    {
      // 20 bytes
      if (convert_hex_literal(the_value, new_expr, 160))
        return true;
      new_expr.type().set("#sol_type", "ADDRESS");
      break;
    }
    default:
      assert(!"Literal not implemented");
    }
    break;
  }
  case SolidityGrammar::ExpressionT::LiteralWithWei:
  case SolidityGrammar::ExpressionT::LiteralWithGwei:
  case SolidityGrammar::ExpressionT::LiteralWithSzabo:
  case SolidityGrammar::ExpressionT::LiteralWithFinney:
  case SolidityGrammar::ExpressionT::LiteralWithEther:
  case SolidityGrammar::ExpressionT::LiteralWithSeconds:
  case SolidityGrammar::ExpressionT::LiteralWithMinutes:
  case SolidityGrammar::ExpressionT::LiteralWithHours:
  case SolidityGrammar::ExpressionT::LiteralWithDays:
  case SolidityGrammar::ExpressionT::LiteralWithWeeks:
  case SolidityGrammar::ExpressionT::LiteralWithYears:
  {
    // e.g. _ESBMC_ether(1);
    assert(expr.contains("subdenomination"));
    std::string unit_name = expr["subdenomination"];

    nlohmann::json node = expr; // do copy
    // remove unit
    //! note that this will leads to "failed to get current contract name" error
    // however, since this can only be int_literal, we should be safe to do so
    node.erase("subdenomination");
    exprt l_expr;
    if (get_expr(node, literal_type, l_expr))
      return true;

    std::string f_name = "_ESBMC_" + unit_name;
    std::string f_id = "c:@F@" + f_name;

    side_effect_expr_function_callt call;
    get_library_function_call_no_args(
      f_name, f_id, unsignedbv_typet(256), location, call);
    call.arguments().push_back(l_expr);

    new_expr = call;
    break;
  }
  case SolidityGrammar::ExpressionT::Tuple:
  {
    // "nodeType": "TupleExpression":
    //    1. InitList: uint[3] x = [1, 2, 3];
    //                         x = [1];  x = [1,2];
    //    2. Operator:
    //        - (x+1) % 2
    //        - if( x && (y || z) )
    //    3. TupleExpr:
    //        - multiple returns: return (x, y);
    //        - swap: (x, y) = (y, x)
    //        - constant: (1, 2)

    if (!expr.contains("components"))
    {
      log_error("Unexpected ast json structure, expecting component");
      abort();
    }
    SolidityGrammar::TypeNameT type =
      SolidityGrammar::get_type_name_t(expr["typeDescriptions"]);

    switch (type)
    {
    // case 1
    case SolidityGrammar::TypeNameT::ArrayTypeName:
    {
      assert(literal_type != nullptr);

      // get elem type
      nlohmann::json elem_literal_type =
        make_array_elementary_type(literal_type);

      // get size
      exprt size;
      size = constant_exprt(
        integer2binary(expr["components"].size(), bv_width(int_type())),
        integer2string(expr["components"].size()),
        int_type());

      // get array type
      typet arr_type;
      if (get_type_description(literal_type, arr_type))
        return true;

      // reallocate array size
      arr_type = array_typet(arr_type.subtype(), size);

      // declare static array tuple
      exprt inits;
      inits = gen_zero(arr_type);
      inits.type().set("#sol_type", "ARRAY_LITERAL");
      inits.type().set("#sol_array_size", size.cformat().as_string());

      // populate array
      int i = 0;
      for (const auto &arg : expr["components"].items())
      {
        exprt init;
        if (get_expr(arg.value(), elem_literal_type, init))
          return true;

        inits.operands().at(i) = init;
        i++;
      }
      inits.id("array");

      // They will be covnerted to an aux array in convert_type_expr() function
      new_expr = inits;
      break;
    }

    // case 3
    case SolidityGrammar::TypeNameT::TupleTypeName: // case 3
    {
      /*
      we assume there are three types of tuple expr:
      0. dump: (x,y);
      1. fixed: (x,y) = (y,x);
      2. function-related: 
          2.1. (x,y) = func();
          2.2. return (x,y);

      case 0:
        1. create a struct type
        2. create a struct type instance
        3. new_expr = instance
        e.g.
        (x , y) ==>
        struct Tuple
        {
          uint x,
          uint y
        };
        Tuple tuple;

      case 1:
        1. add special handling in binary operation.
           when matching struct_expr A = struct_expr B,
           divided into A.operands()[i] = B.operands()[i]
           and populated into a code_block.
        2. new_expr = code_block
        e.g.
        (x, y) = (1, 2) ==>
        {
          tuple.x = 1;
          tuple.y = 2;
        }

      case 2:
        1. when parsing the funciton definition, if the returnParam > 1
           make the function return void instead, and create a struct type
        2. when parsing the return statement, if the return value is a tuple,
           create a struct type instance, do assignments,  and return empty;
        3. when the lhs is tuple and rhs is func_call, get_tuple_instance_expr based 
           on the func_call, and do case 1.
        e.g.
        function test() returns (uint, uint)
        {
          return (1,2);
        }
        ==>
        struct Tuple
        {
          uint x;
          uint y;
        }
        function test()
        {
          Tuple tuple;
          tuple.x = 1;
          tuple.y = 2;
          return;
        }
      */

      if (current_lhsDecl)
      {
        // avoid nested
        assert(!current_rhsDecl);

        // we do not create struct-tuple instance for lhs
        code_blockt _block;
        exprt op = nil_exprt();
        for (auto i : expr["components"])
        {
          if (
            i.contains("typeDescriptions") &&
            get_expr(i, i["typeDescriptions"], op))
            return true;

          _block.operands().push_back(op);
        }
        new_expr = _block;
      }
      else
      {
        // 1. construct struct type
        if (get_tuple_definition(expr))
          return true;

        //2. construct struct_type instance
        if (get_tuple_instance(expr, new_expr))
          return true;
      }

      break;
    }

    // case 2
    default:
    {
      if (get_expr(expr["components"][0], literal_type, new_expr))
        return true;
      break;
    }
    }

    break;
  }
  case SolidityGrammar::ExpressionT::CallOptionsExprClass:
  {
    // e.g.
    // 1.
    // address(tmp).call{gas: 1000000, value: 1 ether}(abi.encodeWithSignature("register(string)", "MyName"));
    // 2.
    // function foo(uint a, uint b) public pure returns (uint) {
    //   return a + b;
    // }
    // function callFoo() public pure returns (uint) {
    //     return foo({a: 1, b: 2});
    // }
    assert(expr.contains("expression"));
    nlohmann::json callee_expr_json = expr["expression"];
    if (SolidityGrammar::is_address_member_call(callee_expr_json))
    {
      if (!is_bound)
      {
        if (get_unbound_expr(expr, current_contractName, new_expr))
          return true;

        symbolt dump;
        get_llc_ret_tuple(dump);
        new_expr = symbol_expr(dump);
      }
      else
      {
        assert(expr.contains("options"));
        // pass the ["options"] to addressmembercall, via literal_type
        if (get_expr(callee_expr_json, expr["options"], new_expr))
          return true;
      }
    }
    else
    {
      if (!is_bound)
      {
        if (get_unbound_expr(expr, current_contractName, new_expr))
          return true;

        break;
      }
      else
      {
        if (!expr.contains("options"))
        {
          log_error("Unsupported CallOptionsExprClass");
          return true;
        }
        // e.g. target.deposit{value: msg.value}();
        exprt memcall;
        // target.deposit
        if (get_expr(callee_expr_json, expr["options"], memcall))
          return true;

        new_expr = memcall;
      }
    }

    break;
  }
  case SolidityGrammar::ExpressionT::CallExprClass:
  {
    side_effect_expr_function_callt call;
    const nlohmann::json &callee_expr_json = expr["expression"];

    // * check if it's a low-level call
    if (SolidityGrammar::is_address_member_call(callee_expr_json))
    {
      log_debug("solidity", "\t\t@@@ got address member call");
      if (get_expr(callee_expr_json, new_expr))
        return true;
      break;
    }

    // * check if it's a solidity built-in function
    if (
      !get_esbmc_builtin_ref(callee_expr_json, new_expr) ||
      !get_sol_builtin_ref(expr, new_expr))
    {
      log_debug("solidity", "\t\t@@@ got builtin function call");
      if (new_expr.id() == "typecast")
      {
        // assume it's a wrap/unwrap
        exprt args;
        if (get_expr(expr["arguments"][0], args))
          return true;
        new_expr.op0() = args;
        break;
      }

      if (new_expr.id() == "sideeffect")
      {
        std::string func_name = new_expr.op0().name().as_string();
        if (
          func_name == "_ESBMC_array_push" || func_name == "_ESBMC_array_pop" ||
          func_name == "_ESBMC_array_length")
          break;
      }

      std::string sol_name = new_expr.type().get("#sol_name").as_string();
      if (sol_name == "revert")
      {
        // Special case: revert
        // insert a bool false as the first argument.
        // drop the rest of params.
        call.function() = new_expr;
        call.type() = to_code_type(new_expr.type()).return_type();
        call.arguments().resize(1);
        call.arguments().at(0) = false_exprt();
      }
      else if (sol_name == "require")
      {
        // Special case: require
        // __ESBMC_assume only handle one param.
        // drop the potential second param.
        exprt single_arg;
        if (get_expr(
              expr["arguments"].at(0),
              expr["arguments"].at(0)["typeDescriptions"],
              single_arg))
          return true;
        call.function() = new_expr;
        call.type() = to_code_type(new_expr.type()).return_type();
        call.arguments().resize(1);
        call.arguments().at(0) = single_arg;
      }
      else
      {
        // other solidity built-in functions
        if (get_library_function_call(
              new_expr, new_expr.type(), empty_json, expr, call))
          return true;
      }

      new_expr = call;
      break;
    }

    // * check if its a call-with-options
    if (
      !expr.contains("name") && callee_expr_json.contains("nodeType") &&
      callee_expr_json["nodeType"] == "FunctionCallOptions")
    {
      if (get_expr(callee_expr_json, new_expr))
        return true;
      break;
    }

    // * check if it's a member access call
    if (
      callee_expr_json.contains("nodeType") &&
      callee_expr_json["nodeType"] == "MemberAccess")
    {
      if (get_expr(callee_expr_json, literal_type, new_expr))
        return true;
      break;
    }

    // wrap it in an ImplicitCastExpr to perform conversion of FunctionToPointerDecay
    nlohmann::json implicit_cast_expr =
      make_implicit_cast_expr(callee_expr_json, "FunctionToPointerDecay");
    exprt callee_expr;
    if (get_expr(implicit_cast_expr, callee_expr))
      return true;

    if (
      callee_expr.is_code() && callee_expr.statement() == "function_call" &&
      callee_expr.op1().name() == "_ESBMC_Nondet_Extcall")
    {
      new_expr = callee_expr;
      return false;
    }

    // * check if it's a struct call
    if (expr.contains("kind") && expr["kind"] == "structConstructorCall")
    {
      log_debug("solidity", "\t\t@@@ got struct constructor call");
      // e.g. Book book = Book('Learn Java', 'TP', 1);
      if (callee_expr.type().id() != irept::id_struct)
        return true;

      typet t = callee_expr.type();
      exprt inits = gen_zero(t);

      int ref_id = callee_expr_json["referencedDeclaration"].get<int>();
      const nlohmann::json &struct_ref = find_decl_ref(src_ast_json, ref_id);
      if (struct_ref == empty_json)
        return true;

      const nlohmann::json members = struct_ref["members"];
      const nlohmann::json args = expr["arguments"];

      // popluate components
      for (size_t i = 0; i < inits.operands().size() && i < args.size(); i++)
      {
        exprt init;
        if (get_expr(args.at(i), members.at(i)["typeDescriptions"], init))
          return true;

        const struct_union_typet::componentt *c =
          &to_struct_type(t).components().at(i);
        typet elem_type = c->type();

        solidity_gen_typecast(ns, init, elem_type);
        inits.operands().at(i) = init;
      }

      new_expr = inits;
      break;
    }

    // funciton call expr
    assert(callee_expr.type().is_code());
    typet type = to_code_type(callee_expr.type()).return_type();

    assert(
      callee_expr_json.contains("referencedDeclaration") &&
      !callee_expr_json["referencedDeclaration"].is_null());
    const auto &decl_ref = find_decl_ref(
      src_ast_json, callee_expr_json["referencedDeclaration"].get<int>());
    std::string node_type = decl_ref["nodeType"].get<std::string>();

    // * check if it's a event, error function call
    if (node_type == "EventDefinition" || node_type == "ErrorDefinition")
    {
      log_debug("solidity", "\t\t@@@ got event/error function call");
      assert(expr.contains("arguments"));
      if (get_library_function_call(callee_expr, type, decl_ref, expr, call))
        return true;
      new_expr = call;
      break;
    }

    // * check if it's the funciton inside library node
    //TODO
    log_debug("solidity", "\t\t@@@ got normal function call");
    // * we had ruled out all the special cases
    // * we now confirm it is called by aother contract inside current contract
    // * func() ==> current_func_this.func(&current_func_this);
    if (get_non_library_function_call(decl_ref, expr, call))
      return true;

    new_expr = call;
    break;
  }
  case SolidityGrammar::ExpressionT::ContractMemberCall:
  {
    // ContractMemberCall
    // - x.setAddress();
    // - x.address();
    // - x.val(); ==> property
    // The later one is quite special, as in Solidity variables behave like functions from the perspective of other contracts.
    // e.g. b._addr is not an address, but a function that returns an address.

    // find the parent json which contains arguments
    const auto &func_call_json = find_last_parent(src_ast_json["nodes"], expr);
    assert(!func_call_json.empty());

    auto callee_expr_json = expr;
    const nlohmann::json &caller_expr_json = callee_expr_json["expression"];
    assert(callee_expr_json.contains("referencedDeclaration"));
    assert(caller_expr_json.contains("referencedDeclaration"));

    side_effect_expr_function_callt call;
    const int contract_var_id =
      caller_expr_json["referencedDeclaration"].get<int>();
    assert(!current_baseContractName.empty());
    const nlohmann::json &base_expr_json =
      find_decl_ref(src_ast_json["nodes"], contract_var_id); // contract

    // contract C{ Base x; x.call();} where base.contractname != current_ContractName;
    // therefore, we need to extract the based contract name
    exprt base;
    std::string base_cname = "";
    if (base_expr_json.empty())
    {
      // assume it's 'this'
      if (contract_var_id < 0 && caller_expr_json["name"] == "this")
      {
        exprt this_expr;
        assert(current_functionDecl);
        if (get_func_decl_this_ref(*current_functionDecl, this_expr))
          return true;
        base = this_expr;

        assert(!current_contractName.empty());
        base_cname = current_contractName;
      }
      else
      {
        log_error("Unexpect base expression");
        return true;
      }
    }
    else
    {
      if (get_var_decl_ref(base_expr_json, true, base))
        return true;
      base_cname = base.type().get("#sol_contract").as_string();
      assert(!base_cname.empty());
    }

    const int member_id = callee_expr_json["referencedDeclaration"].get<int>();
    std::string old = current_baseContractName;
    current_baseContractName = base_cname;
    const nlohmann::json &member_decl_ref =
      find_decl_ref(src_ast_json["nodes"], member_id); // methods or variables
    current_baseContractName = old;

    if (member_decl_ref.empty())
    {
      log_error("cannot find member json node reference");
      return true;
    }

    auto elem_type =
      SolidityGrammar::get_contract_body_element_t(member_decl_ref);
    log_debug(
      "solidity",
      "\t\t@@@ got contrant body element = {}",
      SolidityGrammar::contract_body_element_to_str(elem_type));
    switch (elem_type)
    {
    case SolidityGrammar::VarDecl:
    {
      // e.g. x.data()
      // ==> x.data, where data is a state variable in the contract
      // in Solidity the x.data() is read-only
      exprt comp;
      if (get_var_decl_ref(member_decl_ref, false, comp))
        return true;
      const irep_idt comp_name = comp.name();

      // special checks for mapping
      // e.g. _b.map(0)
      // => map[0] or map_uint_get(ins , 0);
      if (comp.type().get("#sol_type") == "MAPPING")
      {
        assert(func_call_json.contains("arguments"));
        assert(member_decl_ref.contains("typeName"));
        assert(member_decl_ref["typeName"].contains("valueType"));
        exprt pos;
        if (get_expr(
              func_call_json["arguments"][0],
              member_decl_ref["typeName"]["valueType"]["typeDescriptions"],
              pos))
          return true;

        bool is_new_expr = newContractSet.count(base_cname);
        if (is_new_expr)
        {
          if (
            !is_bound && tgt_cnt_set.count(base_cname) > 0 &&
            tgt_cnt_set.size() == 1)
            is_new_expr = false;
        }
        // get key/value type
        typet key_t, value_t;
        std::string key_sol_type, val_sol_type;
        if (get_mapping_key_value_type(
              member_decl_ref, key_t, value_t, key_sol_type, val_sol_type))
        {
          log_error("cannot get mapping key/value type");
          return true;
        }
        gen_mapping_key_typecast(pos, location, key_sol_type);

        if (!is_bound)
        {
          // x.member(); ==> nondet();
          get_nondet_expr(value_t, new_expr);
          break;
        }

        if (!is_new_expr)
        {
          assert(comp.type().is_array());
          //TODO: the index type of ESBMC is limited to unsigned long long
          // we need to extend such limit up to unsignedbv_type(256)
          new_expr = index_exprt(comp, pos, value_t);
        }
        else
        {
          bool is_mapping_set = false;
          auto _mem_call = member_exprt(base, comp.name(), comp.type());
          if (get_new_mapping_index_access(
                value_t,
                val_sol_type,
                is_mapping_set,
                _mem_call,
                pos,
                location,
                new_expr))
            return true;
        }

        break;
      }

      if (current_contractName == base_cname)
        // this.member();
        // comp can be either symbol_expr or member_expr
        new_expr = member_exprt(base, comp_name, comp.type());
      else if (!is_bound)
        // x.member(); ==> nondet();
        get_nondet_expr(comp.type(), new_expr);
      else
      {
        assert(!comp.is_member());
        auto _mem_call = member_exprt(base, comp_name, comp.type());
        if (get_high_level_member_access(
              func_call_json, base, comp, _mem_call, false, new_expr))
          return true;
      }

      break;
    }
    case SolidityGrammar::FunctionDef:
    {
      // e.g. x.func()
      // x    --> base
      // func --> comp
      exprt comp;
      if (get_func_decl_ref(member_decl_ref, comp))
        return true;

      if (get_non_library_function_call(member_decl_ref, func_call_json, call))
        return true;
      call.arguments().at(0) = base;

      if (current_contractName == base_cname)
      {
        // this.init(); we know the implementation thus cannot model it as unbound_harness
        // note that here is comp.identifier not comp.name
        // in unbound mode, we cannot determine the sender
        // wrap with msg_sender update:
        //  old_sender = msg_sender
        //  msg_sender = this.address
        //  ...
        //  msg_sender = old_sender
        // uint160_t old_sender =  msg_sender;
        std::string debug_modulename = get_modulename_from_path(absolute_path);
        exprt msg_sender = symbol_expr(*context.find_symbol("c:@msg_sender"));
        exprt this_expr;
        assert(current_functionDecl);
        if (get_func_decl_this_ref(*current_functionDecl, this_expr))
          return true;

        typet addr_t = unsignedbv_typet(160);
        addr_t.set("#sol_type", "ADDRESS");
        symbolt old_sender;
        get_default_symbol(
          old_sender,
          debug_modulename,
          addr_t,
          "old_sender",
          "sol:@C@" + current_contractName + "@F@old_sender#" +
            std::to_string(aux_counter++),
          locationt());
        symbolt &added_old_sender = *move_symbol_to_context(old_sender);
        code_declt old_sender_decl(symbol_expr(added_old_sender));
        added_old_sender.value = msg_sender;
        old_sender_decl.operands().push_back(msg_sender);
        move_to_front_block(old_sender_decl);

        // msg_sender = this.address;
        exprt this_address = member_exprt(this_expr, "$address", addr_t);
        exprt assign_sender = side_effect_exprt("assign", addr_t);
        assign_sender.copy_to_operands(msg_sender, this_address);
        convert_expression_to_code(assign_sender);
        move_to_front_block(assign_sender);

        // msg_sender = old_sender;
        exprt assign_sender_restore = side_effect_exprt("assign", addr_t);
        assign_sender_restore.copy_to_operands(
          msg_sender, symbol_expr(added_old_sender));
        convert_expression_to_code(assign_sender_restore);
        move_to_back_block(assign_sender_restore);

        new_expr = call;
      }
      else if (!is_bound)
      {
        if (get_unbound_expr(func_call_json, current_contractName, new_expr))
          return true;

        typet t = to_code_type(comp.type()).return_type();
        get_nondet_expr(t, new_expr);
      }
      else
      {
        assert(!comp.is_member());
        if (get_high_level_member_access(
              func_call_json, literal_type, base, comp, call, true, new_expr))
          return true;
      }

      break;
    }
    default:
    {
      log_error(
        "Unexpected Member Access Element Type, Got {}",
        SolidityGrammar::contract_body_element_to_str(elem_type));
      return true;
    }
    }

    break;
  }
  case SolidityGrammar::ExpressionT::ImplicitCastExprClass:
  {
    if (get_cast_expr(expr, new_expr, literal_type))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::IndexAccess:
  {
    const nlohmann::json &base_json = expr["baseExpression"];
    const nlohmann::json &index_json = expr["indexExpression"];

    // 1. get type, this is the base type of array
    typet t;
    if (get_type_description(expr["typeDescriptions"], t))
      return true;

    // 2. get the decl ref of the array
    exprt array;

    // 2.1 arr[n] / x.arr[n]
    if (base_json.contains("referencedDeclaration"))
    {
      if (get_expr(base_json, literal_type, array))
        return true;
    }
    else
    {
      // 2.2 func()[n]
      const nlohmann::json &decl = base_json;
      nlohmann::json implicit_cast_expr =
        make_implicit_cast_expr(decl, "ArrayToPointerDecay");
      if (get_expr(implicit_cast_expr, literal_type, array))
        return true;
    }

    // 3. get the position index
    exprt pos;
    if (get_expr(index_json, expr["typeDescriptions"], pos))
      return true;

    // for MAPPING
    typet base_t;
    if (get_type_description(base_json["typeDescriptions"], base_t))
      return true;
    if (base_t.get("#sol_type").as_string() == "MAPPING")
    {
      // hack to improve the verification speed
      bool is_new_expr = newContractSet.count(current_contractName);
      if (is_new_expr)
      {
        if (
          !is_bound && tgt_cnt_set.count(current_contractName) > 0 &&
          tgt_cnt_set.size() == 1)
          is_new_expr = false;
      }

      // find mapping definition
      assert(base_json.contains("referencedDeclaration"));
      const nlohmann::json &map_node = find_decl_ref(
        src_ast_json["nodes"], base_json["referencedDeclaration"].get<int>());

      // get key/value type
      typet key_t, value_t;
      std::string key_sol_type, val_sol_type;
      if (get_mapping_key_value_type(
            map_node, key_t, value_t, key_sol_type, val_sol_type))
      {
        log_error("cannot get mapping key/value type");
        return true;
      }
      gen_mapping_key_typecast(pos, location, key_sol_type);

      if (!is_new_expr)
      {
        assert(array.type().is_array());
        //TODO: the index type of ESBMC is limited to unsigned long long
        // we need to extend such limit up to unsignedbv_type(256)
        new_expr = index_exprt(array, pos, t);
      }
      else
      {
        bool is_mapping_set = is_mapping_set_lvalue(expr);
        if (get_new_mapping_index_access(
              value_t,
              val_sol_type,
              is_mapping_set,
              array,
              pos,
              location,
              new_expr))
          return true;
      }
      break;
    }

    // for BYTESN, where the index access is read-only
    if (is_bytes_type(t))
    {
      // this means we are dealing with bytes type
      // jump out if it's "bytes[]" or "bytesN[]" or "func()[]"
      SolidityGrammar::TypeNameT tname =
        SolidityGrammar::get_type_name_t(base_json["typeDescriptions"]);
      if (
        !(tname == SolidityGrammar::ArrayTypeName ||
          tname == SolidityGrammar::DynArrayTypeName) &&
        base_json.contains("referencedDeclaration"))
      {
        // e.g.
        //    bytes3 x = 0x123456
        //    bytes1 y = x[0]; // 0x12
        //    bytes1 z = x[1]; // 0x34
        // which equals to
        //    bytes1 z = bswap(x) >> 1 & 0xff
        // for bytes32 x = "test";
        //    x[10] == 0x00 due to the padding
        exprt src_val, src_offset, bswap, bexpr;

        const nlohmann::json &decl = find_decl_ref(
          src_ast_json, base_json["referencedDeclaration"].get<int>());
        if (decl == empty_json)
          return true;

        if (get_var_decl_ref(decl, true, src_val))
          return true;

        if (get_expr(index_json, expr["typeDescriptions"], src_offset))
          return true;

        // extract particular byte based on idx (offset)
        bexpr = exprt("byte_extract_big_endian", src_val.type());
        bexpr.copy_to_operands(src_val, src_offset);

        solidity_gen_typecast(ns, bexpr, unsignedbv_typet(8));

        new_expr = bexpr;
        break;
      }
    }

    // BYTES:  func_ret_bytes()[]
    // same process as above
    if (is_bytes_type(array.type()))
    {
      exprt bexpr = exprt("byte_extract_big_endian", pos.type());
      bexpr.copy_to_operands(array, pos);
      solidity_gen_typecast(ns, bexpr, unsignedbv_typet(8));
      new_expr = bexpr;
      break;
    }

    new_expr = index_exprt(array, pos, t);
    break;
  }
  case SolidityGrammar::ExpressionT::NewExpression:
  {
    // 1. new dynamic array, e.g.
    //    uint[] memory a = new uint[](7);
    //    uint[] memory a = new uint[](len);
    // 2. new bytes array e.g.
    //    bytes memory b = new bytes(7)
    // 3. new object, e.g.
    //    Base x = new Base(1, 2);

    nlohmann::json callee_expr_json = expr["expression"];
    if (callee_expr_json.contains("typeName"))
    {
      // case 1
      // e.g.
      //    new uint[](7)
      // convert to
      //    uint y[7] = {0,0,0,0,0,0,0};
      if (is_dyn_array(callee_expr_json["typeName"]))
      {
        if (get_empty_array_ref(expr, new_expr))
          return true;
        break;
      }
      // case 2
      // the contract/constructor name cannot be "bytes"
      if (
        callee_expr_json["typeName"]["typeDescriptions"]["typeString"]
          .get<std::string>() == "bytes")
      {
        // populate 0x00 to bytes array
        // same process in case SolidityGrammar::ExpressionT::Literal
        assert(expr.contains("arguments") && expr["arguments"].size() == 1);

        int byte_size = stoi(expr["arguments"][0]["value"].get<std::string>());
        std::string hex_val = "";

        for (int i = 0; i < byte_size; i++)
          hex_val += "00";
        hex_val.resize(byte_size * 2);

        if (convert_hex_literal(hex_val, new_expr, byte_size * 8))
          return true;
        break;
      }
    }

    // case 3
    // is equal to Base x = base.base(x);
    exprt call;
    if (get_new_object_ctor_call(expr, call))
      return true;

    new_expr = call;

    if (is_bound)
    {
      // fix: x._ESBMC_bind_cname = Base
      // lhs
      exprt lhs;
      if (get_bind_cname_expr(expr, lhs))
        break; // no lhs

      // rhs
      int ref_decl_id = callee_expr_json["typeName"]["referencedDeclaration"];
      const std::string contract_name = contractNamesMap[ref_decl_id];
      string_constantt rhs(contract_name);

      // do assignment
      exprt _assign = side_effect_exprt("assign", lhs.type());
      solidity_gen_typecast(ns, rhs, lhs.type());
      _assign.operands().push_back(lhs);
      _assign.operands().push_back(rhs);

      convert_expression_to_code(_assign);
      move_to_back_block(_assign);
    }

    break;
  }

  // 1. ContractMemberCall: contractInstance.call()
  //                        contractInstanceArray[0].call()
  //                        contractInstance.x
  //    this should be handled in CallExprClass
  // 2. StructMemberCall: struct.member
  // 3. EnumMemberCall: enum.member
  // 4. AddressMemberCall: tx.origin, msg.sender, ...
  case SolidityGrammar::ExpressionT::StructMemberCall:
  {
    assert(expr.contains("expression"));
    const nlohmann::json caller_expr_json = expr["expression"];

    exprt base;
    if (get_expr(caller_expr_json, base))
      return true;

    const int struct_var_id = expr["referencedDeclaration"].get<int>();
    const nlohmann::json &struct_var_ref =
      find_decl_ref(src_ast_json, struct_var_id);
    if (struct_var_ref == empty_json)
    {
      log_error("cannot find struct member reference");
      return true;
    }
    exprt comp;
    if (get_var_decl_ref(struct_var_ref, true, comp))
      return true;

    assert(comp.name() == expr["memberName"]);
    new_expr = member_exprt(base, comp.name(), comp.type());

    break;
  }
  case SolidityGrammar::ExpressionT::EnumMemberCall:
  {
    assert(expr.contains("expression"));
    const int enum_id = expr["referencedDeclaration"].get<int>();
    const nlohmann::json &enum_member_ref =
      find_decl_ref_unique_id(src_ast_json, enum_id);
    if (enum_member_ref == empty_json)
      return true;

    if (get_enum_member_ref(enum_member_ref, new_expr))
      return true;

    break;
  }
  case SolidityGrammar::ExpressionT::AddressMemberCall:
  {
    // property: <address>.balance
    // function_call: <address>.transfer()
    // examples:
    // 1. address(this).balance;
    // 2.  A tmp = new A();
    //    address(tmp).balance;
    // 3. address x;
    //    x.balance;
    //    msg.sender.balance;
    //! Note that member call like msg.sender will not be handled here
    // The main difference is that, for case 1 we do not need to guess the contract instance
    // While in case 2, we need to utilize over-approximate modelling to bind the all possible instance
    //
    // algo:
    // 1. we add the property and function to the contract definition (not handled here)
    // 2. we create an auxiliary mapping to store the <addr, contract-instance-ptr> pair (not handled here)
    // 3. For case 2, where we only have the address, we need to obtain the object from the mapping
    // For case 1: => this->balance
    // For case 3: => tmp.balance
    const nlohmann::json &caller_expr_json = expr["expression"];
    const std::string mem_name = expr["memberName"].get<std::string>();

    SolidityGrammar::ExpressionT _type =
      SolidityGrammar::get_expression_t(caller_expr_json);
    log_debug(
      "solidity",
      "\t\t@@@ got = {}",
      SolidityGrammar::expression_to_str(_type));

    exprt base;
    switch (_type)
    {
    case SolidityGrammar::TypeConversionExpression:
    case SolidityGrammar::DeclRefExprClass:
    case SolidityGrammar::BuiltinMemberCall:
    {
      // e.g.
      // - address(msg.sender)
      if (get_expr(caller_expr_json, base))
        return true;
      break;
    }
    default:
    {
      log_error(
        "unexpected address member access, got {}",
        SolidityGrammar::expression_to_str(_type));
      return true;
    }
    }

    // case 1, which is a type conversion node
    if (is_low_level_call(mem_name))
    {
      if (!is_bound)
      {
        if (get_unbound_expr(expr, current_contractName, new_expr))
          return true;

        if (mem_name == "send")
          new_expr = nondet_bool_expr;
        else
        {
          // call, staticcall ...
          symbolt dump;
          get_llc_ret_tuple(dump);
          new_expr = symbol_expr(dump);
        }
      }
      else
      {
        const nlohmann::json &initial_func_call =
          find_last_parent(src_ast_json["nodes"], expr);
        const nlohmann::json *func_call = &initial_func_call;

        if ((*func_call)["nodeType"] == "FunctionCallOptions")
        {
          const nlohmann::json &second_call =
            find_last_parent(src_ast_json["nodes"], initial_func_call);
          func_call = &second_call;
        }

        assert((*func_call)["nodeType"] == "FunctionCall");

        if ((*func_call).empty() || (*func_call).is_null())
          return true;

        exprt arg = nil_exprt();
        assert((*func_call).contains("arguments"));

        // only one possible arguemnt
        if ((*func_call)["arguments"].size() > 0)
        {
          auto &arguments = (*func_call)["arguments"][0];
          if (get_expr(arguments, expr["argumentTypes"][0], arg))
            return true;
        }

        if (get_low_level_member_accsss(
              expr, literal_type, mem_name, base, arg, new_expr))
          return true;
      }
    }
    else if (is_low_level_property(mem_name))
    {
      // property i.e balance/codehash
      if (!is_bound)
        // make it as a NODET_UINT
        new_expr = nondet_uint_expr;
      else
        get_builtin_property_expr(
          current_contractName, mem_name, base, location, new_expr);
    }
    else
    {
      log_error("unexpected address member access");
      return true;
    }

    break;
  }
  case SolidityGrammar::ExpressionT::LibraryMemberCall:
  {
    side_effect_expr_function_callt call;
    const auto &func_ref = find_decl_ref_unique_id(
      src_ast_json, expr["referencedDeclaration"].get<int>());

    const nlohmann::json &args_json =
      find_last_parent(src_ast_json["nodes"], expr);
    assert(args_json.contains("arguments"));
    if (get_library_function_call(func_ref, args_json, call))
      return true;

    exprt base;
    const nlohmann::json caller_expr_json = expr["expression"];
    if (get_expr(caller_expr_json, literal_type, base))
      return true;

    if (!(base.is_code() && base.type().get("#sol_type") == "LIBRARY"))
      // this means it is a using for library
      call.arguments().insert(call.arguments().begin(), base);

    new_expr = call;
    break;
  }
  case SolidityGrammar::ExpressionT::BuiltinMemberCall:
  {
    if (get_sol_builtin_ref(expr, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::TypePropertyExpression:
  {
    // e.g.
    // Integers: 'type(uint256)'.max/min
    // Contracts: 'type(MyContract)'.creationCode/runtimecode

    // create a dump expr with no value, but set the correct type
    assert(
      expr.contains("expression") &&
      expr["expression"].contains("argumentTypes"));
    const auto &json = expr["expression"]["argumentTypes"];

    /*
    e.g.
      "argumentTypes": [
          {
              "typeIdentifier": "t_type$_t_int256_$",
              "typeString": "type(int256)"
          }
      ],
    */
    typet t;
    if (get_type_description(json[0], t))
      return true;

    exprt dump;
    dump.type() = t;

    new_expr = dump;
    break;
  }
  case SolidityGrammar::ExpressionT::TypeConversionExpression:
  {
    // perform type conversion
    // e.g.
    // address payable public owner = payable(msg.sender);
    // or
    // uint32 a = 0x432178;
    // uint16 b = uint16(a); // b will be 0x2178 now

    assert(expr.contains("expression"));
    const nlohmann::json conv_expr = expr["expression"];
    typet type;
    exprt from_expr;

    // 1. get source expr
    //! assume: only one argument
    assert(expr["arguments"].size() == 1);
    if (get_expr(expr["arguments"][0], literal_type, from_expr))
      return true;

    // 2. get target type
    if (get_type_description(conv_expr["typeDescriptions"], type))
      return true;

    // 3. generate the type casting expr
    convert_type_expr(ns, from_expr, type);

    new_expr = from_expr;
    break;
  }
  case SolidityGrammar::ExpressionT::NullExpr:
  {
    // e.g. (, x) = (1, 2);
    // the first component in lhs is nil
    new_expr = nil_exprt();
    break;
  }
  default:
  {
    assert(!"Unimplemented type in rule expression");
    return true;
  }
  }

  new_expr.location() = location;

  log_debug(
    "solidity",
    "@@@ Finish parsing expresion = {}",
    SolidityGrammar::expression_to_str(type));
  return false;
}

// get the initial value for the variable declaration
bool solidity_convertert::get_init_expr(
  const nlohmann::json &init_value,
  const nlohmann::json &literal_type,
  const typet &dest_type,
  exprt &new_expr)
{
  if (literal_type == nullptr)
    return true;

  if (get_expr(init_value, literal_type, new_expr))
    return true;

  convert_type_expr(ns, new_expr, dest_type);
  return false;
}

// get the name of the contract that contains the target ast_node, including library
// note that the contract_name might be empty
void solidity_convertert::get_current_contract_name(
  const nlohmann::json &ast_node,
  std::string &contract_name)
{
  log_debug("solidity", "\tfinding current contract name");
  if (ast_node.is_null() || ast_node.empty())
  {
    log_debug("solidity", "got empty contract name");
    contract_name = "";
    return;
  }
  if (!ast_node.contains("id"))
  {
    // this could be manually created json.
    //TODO: avoid this kind of implementation
    if (ast_node.is_object() && ast_node["nodeType"] == "ImplicitCastExprClass")
    {
      get_current_contract_name(ast_node["subExpr"], contract_name);
    }
    else
    {
      log_warning("target node do not have id.");
      if (!ast_node.is_object())
        abort();
      log_status("{}", ast_node.dump());
    }
    return;
  }

  const auto &json = find_parent_contract(src_ast_json["nodes"], ast_node);
  if (json.empty() || json.is_null())
  {
    log_debug(
      "solidity",
      "failed to get current contract name, trying to "
      "find id {}, target json is \n{}\n",
      std::to_string(ast_node["id"].get<int>()),
      ast_node.dump());
    return;
  }

  assert(json.contains("name"));

  contract_name = json["name"].get<std::string>();
  log_debug("solidity", "\tcurrent contract name={}", contract_name);
}

bool solidity_convertert::get_binary_operator_expr(
  const nlohmann::json &expr,
  exprt &new_expr)
{
  // preliminary step for recursive BinaryOperation
  current_BinOp_type.push(&(expr["typeDescriptions"]));

  // 1. Convert LHS and RHS
  // For "Assignment" expression, it's called "leftHandSide" or "rightHandSide".
  // For "BinaryOperation" expression, it's called "leftExpression" or "leftExpression"
  exprt lhs, rhs;
  nlohmann::json rhs_json;
  locationt l;
  get_location_from_node(expr, l);

  if (expr.contains("leftHandSide"))
  {
    nlohmann::json literalType = expr["leftHandSide"]["typeDescriptions"];

    current_lhsDecl = true;
    if (get_expr(expr["leftHandSide"], lhs))
      return true;
    current_lhsDecl = false;

    current_rhsDecl = true;
    if (get_expr(expr["rightHandSide"], literalType, rhs))
      return true;
    current_rhsDecl = false;

    rhs_json = expr["rightHandSide"];
  }
  else if (expr.contains("leftExpression"))
  {
    nlohmann::json literalType_l = expr["leftExpression"]["typeDescriptions"];
    nlohmann::json literalType_r = expr["rightExpression"]["typeDescriptions"];

    current_lhsDecl = true;
    if (get_expr(expr["leftExpression"], literalType_l, lhs))
      return true;
    current_lhsDecl = false;

    current_rhsDecl = true;
    if (get_expr(expr["rightExpression"], literalType_r, rhs))
      return true;
    current_rhsDecl = false;

    rhs_json = expr["rightExpression"];
  }
  else
    assert(!"should not be here - unrecognized LHS and RHS keywords in expression JSON");

  // 2. Get type
  typet t;
  assert(current_BinOp_type.size());
  const nlohmann::json &binop_type = *(current_BinOp_type.top());
  if (get_type_description(binop_type, t))
    return true;

  typet common_type;
  if (expr.contains("commonType"))
  {
    if (get_type_description(expr["commonType"], common_type))
      return true;
  }

  typet lt = lhs.type();
  typet rt = rhs.type();
  std::string lt_sol = lt.get("#sol_type").as_string();
  std::string rt_sol = rt.get("#sol_type").as_string();

  // 2.1 special handling for the sol_unbound harness
  convert_unboundcall_nondet(lhs, common_type, l);
  convert_unboundcall_nondet(rhs, common_type, l);

  // 3. Convert opcode
  SolidityGrammar::ExpressionT opcode =
    SolidityGrammar::get_expr_operator_t(expr);
  log_debug(
    "solidity",
    "	@@@ got binop.getOpcode: SolidityGrammar::{}",
    SolidityGrammar::expression_to_str(opcode));

  switch (opcode)
  {
  case SolidityGrammar::ExpressionT::BO_Assign:
  {
    // special handling for tuple-type assignment;
    //TODO: handle nested tuple
    if (rt_sol == "TUPLE_INSTANCE" || rt_sol == "TUPLE_RETURNS")
    {
      construct_tuple_assigments(expr, lhs, rhs);
      new_expr = code_skipt();
      current_BinOp_type.pop();
      return false;
    }
    else if (rt_sol == "ARRAY" || rt_sol == "ARRAY_LITERAL")
    {
      if (rt_sol == "ARRAY_LITERAL")
        // construct aux_array while adding padding
        // e.g. data1 = [1,2] ==> data1 = aux_array$1
        convert_type_expr(ns, rhs, lt);

      // get size
      exprt size_expr;
      get_size_expr(rhs, size_expr);

      // get sizeof
      exprt size_of_expr;
      get_size_of_expr(rt.subtype(), size_of_expr);

      // do array copy
      side_effect_expr_function_callt acpy_call;
      get_arrcpy_function_call(lhs.location(), acpy_call);
      acpy_call.arguments().push_back(rhs);
      acpy_call.arguments().push_back(size_expr);
      acpy_call.arguments().push_back(size_of_expr);
      solidity_gen_typecast(ns, acpy_call, lt);

      rhs = acpy_call;

      if (lt_sol == "DYNARRAY")
      {
        exprt store_call;
        store_update_dyn_array(lhs, size_expr, store_call);
        move_to_back_block(store_call);
      }
    }
    else if (rt_sol == "DYNARRAY")
    {
      /* e.g. 
        int[] public data1;
        int[] memory ac;
        ac = new int[](10);
        data1 = ac;  // target

      we convert it as 
        data1 = _ESBMC_arrcpy(ac, get_array_size(ac), type_size); 
        _ESBMC_store_array(data1, ac_size);
      */

      // get size
      exprt size_expr;
      get_size_expr(rhs, size_expr);

      // get sizeof
      exprt size_of_expr;
      get_size_of_expr(rt.subtype(), size_of_expr);

      // do array copy
      side_effect_expr_function_callt acpy_call;
      get_arrcpy_function_call(lhs.location(), acpy_call);
      acpy_call.arguments().push_back(rhs);
      acpy_call.arguments().push_back(size_expr);
      acpy_call.arguments().push_back(size_of_expr);
      solidity_gen_typecast(ns, acpy_call, lt);

      rhs = acpy_call;

      if (lt_sol == "DYNARRAY")
      {
        exprt store_call;
        store_update_dyn_array(lhs, size_expr, store_call);
        move_to_back_block(store_call);
      }
      // fall through to do assignment
    }
    else if (rt_sol == "NEW_ARRAY")
    {
      /* e.g. 
        int[] public data1;
        int[] memory ac;
        ac = new int[](10);

      we convert it as 
        ac = new int[](10);
        _ESBMC_store_array(ac, ac_size); 
        */
      exprt size_expr;
      if (!rhs_json.contains("arguments"))
        abort();
      nlohmann::json callee_arg_json = rhs_json["arguments"][0];
      const nlohmann::json literal_type = callee_arg_json["typeDescriptions"];
      if (get_expr(callee_arg_json, literal_type, size_expr))
        return true;

      // get sizeof
      exprt size_of_expr;
      get_size_of_expr(rt.subtype(), size_of_expr);

      // do array copy
      side_effect_expr_function_callt acpy_call;
      get_arrcpy_function_call(lhs.location(), acpy_call);
      acpy_call.arguments().push_back(rhs);
      acpy_call.arguments().push_back(size_expr);
      acpy_call.arguments().push_back(size_of_expr);
      solidity_gen_typecast(ns, acpy_call, lt);

      rhs = acpy_call;

      if (lt_sol == "DYNARRAY")
      {
        exprt store_call;
        store_update_dyn_array(lhs, size_expr, store_call);
        move_to_back_block(store_call);
      }
    }
    else if (lt_sol == "STRING")
    {
      get_string_assignment(lhs, rhs, new_expr);
      return false;
    }

    new_expr = side_effect_exprt("assign", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Add:
  {
    if (t.is_floatbv())
      assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
    else
      new_expr = exprt("+", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Sub:
  {
    if (t.is_floatbv())
      assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
    else
      new_expr = exprt("-", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Mul:
  {
    if (t.is_floatbv())
      assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
    else
      new_expr = exprt("*", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Div:
  {
    if (t.is_floatbv())
      assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
    else
      new_expr = exprt("/", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Rem:
  {
    new_expr = exprt("mod", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Shl:
  {
    new_expr = exprt("shl", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Shr:
  {
    new_expr = exprt("shr", t);
    break;
  }
  case SolidityGrammar::BO_And:
  {
    new_expr = exprt("bitand", t);
    break;
  }
  case SolidityGrammar::BO_Xor:
  {
    new_expr = exprt("bitxor", t);
    break;
  }
  case SolidityGrammar::BO_Or:
  {
    new_expr = exprt("bitor", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_GT:
  {
    new_expr = exprt(">", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_LT:
  {
    new_expr = exprt("<", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_GE:
  {
    new_expr = exprt(">=", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_LE:
  {
    new_expr = exprt("<=", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_NE:
  {
    new_expr = exprt("notequal", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_EQ:
  {
    new_expr = exprt("=", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_LAnd:
  {
    new_expr = exprt("and", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_LOr:
  {
    new_expr = exprt("or", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Pow:
  {
    // lhs**rhs => pow(lhs, rhs)

    // optimization: if both base and exponent are constant, use bigint::power
    exprt new_lhs = lhs;
    exprt new_rhs = rhs;
    // remove typecast
    while (lhs.id() == "typecast")
      new_lhs = new_lhs.op0();
    while (rhs.id() == "typecast")
      new_rhs = new_rhs.op0();
    if (new_lhs.is_constant() && new_rhs.is_constant())
    {
      //? it seems the solc cannot generate ast_json for constant power like 2**20
      BigInt base;
      if (to_integer(new_lhs, base))
      {
        log_error("failed to convert constant: {}", new_lhs.pretty());
        abort();
      }

      BigInt exp;
      if (to_integer(new_rhs, exp))
      {
        log_error("failed to convert constant: {}", new_rhs.pretty());
        abort();
      }

      BigInt res = ::power(base, exp);
      exprt tmp = from_integer(res, unsignedbv_typet(256));

      if (tmp.is_nil())
        return true;

      new_expr.swap(tmp);
    }
    else
    {
      // Not sure why but it seems esbmc's pow works worse in solidity,
      // so I write my own version
      //? maybe this should convert this to BigInt too?
      side_effect_expr_function_callt call_expr;
      get_library_function_call_no_args(
        "pow", "c:@F@pow", double_type(), lhs.location(), call_expr);

      call_expr.arguments().push_back(typecast_exprt(lhs, double_type()));
      call_expr.arguments().push_back(typecast_exprt(rhs, double_type()));

      new_expr = call_expr;
    }
    new_expr.location() = l;
    // do not fall through
    return false;
  }
  default:
  {
    if (get_compound_assign_expr(expr, lhs, rhs, common_type, new_expr))
    {
      assert(!"Unimplemented binary operator");
      return true;
    }

    current_BinOp_type.pop();

    return false;
  }
  }

  // for bytes type
  if (is_bytes_type(lhs.type()) || is_bytes_type(rhs.type()))
  {
    switch (opcode)
    {
    case SolidityGrammar::ExpressionT::BO_GT:
    case SolidityGrammar::ExpressionT::BO_LT:
    case SolidityGrammar::ExpressionT::BO_GE:
    case SolidityGrammar::ExpressionT::BO_LE:
    case SolidityGrammar::ExpressionT::BO_NE:
    case SolidityGrammar::ExpressionT::BO_EQ:
    {
      // e.g. cmp(0x74,  0x7500)
      // ->   cmp(0x74, 0x0075)

      // convert string to bytes
      // e.g.
      //    data1 = "test"; data2 = 0x74657374; // "test"
      //    assert(data1 == data2); // true
      // Do type conversion before the bswap
      // the arguement of bswap should only be int/uint type, not string
      // e.g. data1 == "test", it should not be bswap("test")
      // instead it should be bswap(0x74657374)
      // this may overwrite the lhs & rhs.
      if (!is_bytes_type(lhs.type()))
      {
        if (get_expr(expr["leftExpression"], expr["commonType"], lhs))
          return true;
      }
      if (!is_bytes_type(rhs.type()))
      {
        if (get_expr(expr["rightExpression"], expr["commonType"], rhs))
          return true;
      }

      // do implicit type conversion
      convert_type_expr(ns, lhs, common_type);
      convert_type_expr(ns, rhs, common_type);

      exprt bwrhs, bwlhs;
      bwlhs = exprt("bswap", common_type);
      bwlhs.operands().push_back(lhs);
      lhs = bwlhs;

      bwrhs = exprt("bswap", common_type);
      bwrhs.operands().push_back(rhs);
      rhs = bwrhs;

      new_expr.copy_to_operands(lhs, rhs);
      // Pop current_BinOp_type.push as we've finished this conversion
      current_BinOp_type.pop();
      return false;
    }
    case SolidityGrammar::ExpressionT::BO_Shl:
    {
      // e.g.
      //    bytes1 = 0x11
      //    x<<8 == 0x00
      new_expr.copy_to_operands(lhs, rhs);
      convert_type_expr(ns, new_expr, lhs.type());
      current_BinOp_type.pop();
      return false;
    }
    default:
    {
      break;
    }
    }
  }

  // 4.1 check if it needs implicit type conversion
  if (common_type.id() != "")
  {
    convert_type_expr(ns, lhs, common_type);
    convert_type_expr(ns, rhs, common_type);
  }

  // 4.2 Copy to operands
  new_expr.copy_to_operands(lhs, rhs);

  // Pop current_BinOp_type.push as we've finished this conversion
  current_BinOp_type.pop();

  return false;
}

bool solidity_convertert::get_compound_assign_expr(
  const nlohmann::json &expr,
  exprt &lhs,
  exprt &rhs,
  typet &common_type,
  exprt &new_expr)
{
  // equivalent to clang_c_convertert::get_compound_assign_expr
  SolidityGrammar::ExpressionT opcode =
    SolidityGrammar::get_expr_operator_t(expr);

  switch (opcode)
  {
  case SolidityGrammar::ExpressionT::BO_AddAssign:
  {
    new_expr = side_effect_exprt("assign+");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_SubAssign:
  {
    new_expr = side_effect_exprt("assign-");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_MulAssign:
  {
    new_expr = side_effect_exprt("assign*");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_DivAssign:
  {
    new_expr = side_effect_exprt("assign_div");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_RemAssign:
  {
    new_expr = side_effect_exprt("assign_mod");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_ShlAssign:
  {
    new_expr = side_effect_exprt("assign_shl");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_ShrAssign:
  {
    new_expr = side_effect_exprt("assign_shr");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_AndAssign:
  {
    new_expr = side_effect_exprt("assign_bitand");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_XorAssign:
  {
    new_expr = side_effect_exprt("assign_bitxor");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_OrAssign:
  {
    new_expr = side_effect_exprt("assign_bitor");
    break;
  }
  default:
    return true;
  }

  if (common_type.id() != "")
  {
    convert_type_expr(ns, lhs, common_type);
    convert_type_expr(ns, rhs, common_type);
  }

  new_expr.copy_to_operands(lhs, rhs);
  return false;
}

bool solidity_convertert::get_unary_operator_expr(
  const nlohmann::json &expr,
  const nlohmann::json &literal_type,
  exprt &new_expr)
{
  // TODO: Fix me! Currently just support prefix == true,e.g. pre-increment

  // 1. get UnaryOperation opcode
  SolidityGrammar::ExpressionT opcode =
    SolidityGrammar::get_unary_expr_operator_t(expr, expr["prefix"]);
  log_debug(
    "solidity",
    "	@@@ got uniop.getOpcode: SolidityGrammar::{}",
    SolidityGrammar::expression_to_str(opcode));

  // 2. get type
  typet uniop_type;
  if (get_type_description(expr["typeDescriptions"], uniop_type))
    return true;

  // 3. get subexpr
  exprt unary_sub;
  if (get_expr(expr["subExpression"], literal_type, unary_sub))
    return true;

  switch (opcode)
  {
  case SolidityGrammar::ExpressionT::UO_PreDec:
  {
    new_expr = side_effect_exprt("predecrement", uniop_type);
    break;
  }
  case SolidityGrammar::ExpressionT::UO_PreInc:
  {
    new_expr = side_effect_exprt("preincrement", uniop_type);
    break;
  }
  case SolidityGrammar::UO_PostDec:
  {
    new_expr = side_effect_exprt("postdecrement", uniop_type);
    break;
  }
  case SolidityGrammar::UO_PostInc:
  {
    new_expr = side_effect_exprt("postincrement", uniop_type);
    break;
  }
  case SolidityGrammar::ExpressionT::UO_Minus:
  {
    new_expr = exprt("unary-", uniop_type);
    break;
  }
  case SolidityGrammar::ExpressionT::UO_Not:
  {
    new_expr = exprt("bitnot", uniop_type);
    break;
  }

  case SolidityGrammar::ExpressionT::UO_LNot:
  {
    new_expr = exprt("not", bool_type());
    break;
  }
  default:
  {
    assert(!"Unimplemented unary operator");
  }
  }

  new_expr.operands().push_back(unary_sub);
  return false;
}

bool solidity_convertert::get_conditional_operator_expr(
  const nlohmann::json &expr,
  exprt &new_expr)
{
  exprt cond;
  if (get_expr(expr["condition"], cond))
    return true;

  exprt then;
  if (get_expr(expr["trueExpression"], expr["typeDescriptions"], then))
    return true;

  exprt else_expr;
  if (get_expr(expr["falseExpression"], expr["typeDescriptions"], else_expr))
    return true;

  typet t;
  if (get_type_description(expr["typeDescriptions"], t))
    return true;

  exprt if_expr("if", t);
  if_expr.copy_to_operands(cond, then, else_expr);

  new_expr = if_expr;

  return false;
}

bool solidity_convertert::get_cast_expr(
  const nlohmann::json &cast_expr,
  exprt &new_expr,
  const nlohmann::json literal_type)
{
  // 1. convert subexpr
  exprt expr;
  if (get_expr(cast_expr["subExpr"], literal_type, expr))
    return true;

  // 2. get type
  typet type;
  if (cast_expr["castType"].get<std::string>() == "ArrayToPointerDecay")
  {
    // Array's cast_expr will have cast_expr["subExpr"]["typeDescriptions"]:
    //  "typeIdentifier": "t_array$_t_uint8_$2_memory_ptr"
    //  "typeString": "uint8[2] memory"
    // For the data above, SolidityGrammar::get_type_name_t will return ArrayTypeName.
    // But we want Pointer type. Hence, adjusting the type manually to make it like:
    //   "typeIdentifier": "ArrayToPtr",
    //   "typeString": "uint8[2] memory"
    nlohmann::json adjusted_type =
      make_array_to_pointer_type(cast_expr["subExpr"]["typeDescriptions"]);
    if (get_type_description(adjusted_type, type))
      return true;
  }
  // TODO: Maybe can just type = expr.type() for other types as well. Need to make sure types are all set in get_expr (many functions are called multiple times to perform the same action).
  else
  {
    type = expr.type();
  }

  // 3. get cast type and generate typecast
  SolidityGrammar::ImplicitCastTypeT cast_type =
    SolidityGrammar::get_implicit_cast_type_t(
      cast_expr["castType"].get<std::string>());
  switch (cast_type)
  {
  case SolidityGrammar::ImplicitCastTypeT::LValueToRValue:
  {
    solidity_gen_typecast(ns, expr, type);
    break;
  }
  case SolidityGrammar::ImplicitCastTypeT::FunctionToPointerDecay:
  case SolidityGrammar::ImplicitCastTypeT::ArrayToPointerDecay:
  {
    break;
  }
  default:
  {
    assert(!"Unimplemented implicit cast type");
  }
  }

  new_expr = expr;
  return false;
}

// always success
void solidity_convertert::get_symbol_decl_ref(
  const std::string &sym_name,
  const std::string &sym_id,
  const typet &t,
  exprt &new_expr)
{
  if (context.find_symbol(sym_id) != nullptr)
    new_expr = symbol_expr(*context.find_symbol(sym_id));
  else
  {
    new_expr = exprt("symbol", t);
    new_expr.identifier(sym_id);
    new_expr.cmt_lvalue(true);
    new_expr.name(sym_name);
    new_expr.pretty_name(sym_name);
  }
}

/**
  This function can return expr with either id::symbol or id::member
  id::memebr can only be the case where this.xx
  @decl: declaration json node
  @is_this_ptr: whether we need to convert x => this.x
  @cname: based contract name
*/
bool solidity_convertert::get_var_decl_ref(
  const nlohmann::json &decl,
  const bool is_this_ptr,
  exprt &new_expr)
{
  // Function to configure new_expr that has a +ve referenced id, referring to a variable declaration
  assert(decl["nodeType"] == "VariableDeclaration");
  std::string name, id;
  if (get_var_decl_name(decl, name, id))
    return true;

  typet type;
  const nlohmann::json *old_typeName = current_typeName;
  current_typeName = &decl["typeName"];
  if (get_type_description(decl["typeName"]["typeDescriptions"], type))
    return true;
  current_typeName = old_typeName;

  bool is_global_static_mapping =
    type.get("#sol_type") == "MAPPING" && type.is_array();

  if (context.find_symbol(id) != nullptr)
    new_expr = symbol_expr(*context.find_symbol(id));
  else
  {
    // variable with no value
    new_expr = exprt("symbol", type);
    new_expr.identifier(id);
    new_expr.cmt_lvalue(true);
    new_expr.name(name);
    new_expr.pretty_name(name);
  }

  if (is_this_ptr && !is_global_static_mapping)
  {
    if (decl["stateVariable"] && current_functionDecl)
    {
      // check if it's a constant in the library,
      // if so, no need to add the this pointer
      std::string c_name;
      get_current_contract_name(decl, c_name);
      if (!c_name.empty() && contractNamesList.count(c_name) == 0)
      {
        assert(decl["mutability"] == "constant");
        return false;
      }

      // this means we are parsing function body
      // and the variable is a state var
      // data = _data ==> this->data = _data;

      // get function this pointer
      exprt this_ptr;
      assert(current_functionDecl != nullptr);
      if (get_func_decl_this_ref(*current_functionDecl, this_ptr))
        return true;

      // construct member access this->data
      assert(!new_expr.name().empty());
      new_expr = member_exprt(this_ptr, new_expr.name(), new_expr.type());
    }
  }
  return false;
}

/*
  we got two get_func_decl_ref_type,
  this one is to get the expr
*/
bool solidity_convertert::get_func_decl_ref(
  const nlohmann::json &decl,
  exprt &new_expr)
{
  // Function to configure new_expr that has a +ve referenced id, referring to a function declaration
  // This allow to get func symbol before we add it to the symbol table
  assert(
    decl["nodeType"] == "FunctionDefinition" ||
    decl["nodeType"] == "EventDefinition" ||
    decl["nodeType"] == "ErrorDefinition");

  // find base contract name
  const auto contract_def = find_parent_contract(src_ast_json, decl);
  assert(contract_def.contains("name"));
  const std::string cname = contract_def["name"].get<std::string>();

  std::string name, id;
  get_function_definition_name(decl, name, id);

  if (context.find_symbol(id) != nullptr)
  {
    new_expr = symbol_expr(*context.find_symbol(id));
    return false;
  }

  typet type;
  if (get_func_decl_ref_type(
        decl, type)) // "type-name" as in state-variable-declaration
    return true;

  //! function with no value i.e function body
  new_expr = exprt("symbol", type);
  new_expr.identifier(id);
  new_expr.cmt_lvalue(true);
  new_expr.name(name);
  return false;
}

/*
  we got two get_func_decl_ref_type,
  this one is to get the json
  return empty_json if it's not found
*/
const nlohmann::json &solidity_convertert::get_func_decl_ref(
  const std::string &c_name,
  const std::string &f_name)
{
  nlohmann::json &nodes = src_ast_json["nodes"];
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    if ((*itr)["nodeType"] == "ContractDefinition" && (*itr)["name"] == c_name)
    {
      nlohmann::json &ast_nodes = (*itr)["nodes"];
      for (nlohmann::json::iterator itrr = ast_nodes.begin();
           itrr != ast_nodes.end();
           ++itrr)
      {
        if ((*itrr)["nodeType"] == "FunctionDefinition")
        {
          if ((*itrr).contains("name") && (*itrr)["name"] == f_name)
            return (*itrr);
          if ((*itrr).contains("kind") && (*itrr)["kind"] == f_name)
            return (*itrr);
        }
      }
    }
  }

  return empty_json;
}

// wrapper
bool solidity_convertert::get_func_decl_this_ref(
  const nlohmann::json &decl,
  exprt &new_expr)
{
  assert(
    !decl.empty() &&
    !decl.is_null()); // yet we cannot detect if it's a Dangling
  std::string func_name, func_id;
  get_function_definition_name(decl, func_name, func_id);
  std::string current_contractName;
  get_current_contract_name(decl, current_contractName);
  if (current_contractName.empty())
  {
    log_error("failed to obtain current contract name");
    return true;
  }

  return get_func_decl_this_ref(current_contractName, func_id, new_expr);
}

// get the this pointer symbol
bool solidity_convertert::get_func_decl_this_ref(
  const std::string contract_name,
  const std::string &func_id,
  exprt &new_expr)
{
  log_debug(
    "solidity",
    "\t@@@ get this reference of func {} in contract {}",
    func_id,
    contract_name);
  std::string this_id = func_id + "#this";
  locationt l;
  code_typet type;
  type.return_type() = empty_typet();
  type.return_type().set("cpp_type", "void");

  if (context.find_symbol(this_id) == nullptr)
  {
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    get_function_this_pointer_param(
      contract_name, func_id, debug_modulename, l, type);
  }

  assert(context.find_symbol(this_id) != nullptr);
  new_expr = symbol_expr(*context.find_symbol(this_id));
  return false;
}

bool solidity_convertert::get_ctor_decl_this_ref(
  const nlohmann::json &caller,
  exprt &this_object)
{
  log_debug("solidity", "get_ctor_decl_this_ref");

  // ctor
  std::string current_cname;
  get_current_contract_name(caller, current_cname);
  if (current_cname.empty())
  {
    log_error("Failed to get caller's contract name");
    return true;
  }
  std::string ctor_id;
  if (get_ctor_call_id(current_cname, ctor_id))
  {
    log_error("failed to get the ctor id");
    return true;
  }

  if (get_func_decl_this_ref(current_cname, ctor_id, this_object))
  {
    log_error("failed to get this ref of function {}", ctor_id);
    return true;
  }
  return false;
}

bool solidity_convertert::get_enum_member_ref(
  const nlohmann::json &decl,
  exprt &new_expr)
{
  assert(decl["nodeType"] == "EnumValue");
  assert(decl.contains("Value"));

  const std::string rhs = decl["Value"].get<std::string>();

  new_expr = constant_exprt(
    integer2binary(string2integer(rhs), bv_width(int_type())), rhs, int_type());

  return false;
}

// get the esbmc built-in methods
bool solidity_convertert::get_esbmc_builtin_ref(
  const nlohmann::json &decl,
  exprt &new_expr)
{
  log_debug("solidity", "\t@@@ get_esbmc_builtin_ref");
  // Function to configure new_expr that has a -ve referenced id
  // -ve ref id means built-in functions or variables.
  // Add more special function names here
  if (
    decl.contains("referencedDeclaration") &&
    !decl["referencedDeclaration"].is_null() &&
    decl["referencedDeclaration"].get<int>() >= 0)
    return true;

  if (!decl.contains("name"))
    return get_sol_builtin_ref(decl, new_expr);

  const std::string blt_name = decl["name"].get<std::string>();
  std::string name, id;

  // "require" keyword is virtually identical to "assume"
  if (
    blt_name == "require" || blt_name == "revert" ||
    blt_name == "__ESBMC_assume" || blt_name == "__VERIFIER_assume")
    name = "__ESBMC_assume";
  else if (
    blt_name == "assert" || blt_name == "__ESBMC_assert" ||
    blt_name == "__VERIFIER_assert")
    name = "assert";
  else
    //!assume it's a solidity built-in func
    return get_sol_builtin_ref(decl, new_expr);
  id = "c:@F@" + name;

  if (name == "__ESBMC_assume")
  {
    assert(context.find_symbol(id) != nullptr);
    new_expr = symbol_expr(*context.find_symbol(id));
    new_expr.type().set("#sol_name", blt_name);
  }
  else
  {
    // assert
    typet type;
    code_typet convert_type;
    typet return_type;
    return_type = bool_type();
    std::string c_type = "bool";
    return_type.set("#cpp_type", c_type);
    convert_type.return_type() = return_type;
    type = convert_type;
    type.set("#sol_name", blt_name);

    new_expr = exprt("symbol", type);
    new_expr.identifier(id);
    new_expr.cmt_lvalue(true);
    new_expr.name(name);

    locationt loc;
    get_location_from_node(decl, loc);
    new_expr.location() = loc;
    if (current_functionDecl)
      new_expr.location().function(current_functionName);
  }

  return false;
}

/*
  check if it's a solidity built-in function
  - if so, get the function definition reference, assign to new_expr and return false  
  - if not, return true
*/
bool solidity_convertert::get_sol_builtin_ref(
  const nlohmann::json expr,
  exprt &new_expr)
{
  log_debug(
    "solidity",
    "\t@@@ expecting solidity builtin ref, got nodeType={}",
    expr["nodeType"].get<std::string>());
  // get the reference from the pre-populated symbol table
  // note that this could be either vars or funcs.
  assert(expr.contains("nodeType"));
  locationt l;
  get_location_from_node(expr, l);

  if (expr["nodeType"].get<std::string>() == "FunctionCall")
  {
    //  e.g. gasleft() <=> c:@F@gasleft
    if (expr["expression"]["nodeType"].get<std::string>() != "Identifier")
      // this means it's not a builtin funciton
      return true;

    std::string name = expr["expression"]["name"].get<std::string>();
    std::string id = "c:@F@" + name;
    if (context.find_symbol(id) == nullptr)
      return true;
    const symbolt &sym = *context.find_symbol(id);
    new_expr = symbol_expr(sym);
  }
  else if (expr["nodeType"].get<std::string>() == "MemberAccess")
  {
    // e.g. string.concat() <=> c:@string_concat
    std::string bs;
    if (expr.contains("memberName"))
    {
      // assume it's the no-basename type
      // e.g. address(this).balance, type(uint256).max
      std::string name = expr["memberName"];
      if (name == "max" || name == "min")
      {
        exprt dump;
        if (get_expr(expr["expression"], dump))
          return true;
        std::string sol_str = dump.type().get("#sol_type").as_string();
        // extract integer width: e.g. uint8 => uint + 8
        std::string type = (sol_str[0] == 'U') ? "UINT" : "INT";
        std::string width = sol_str.substr(type.size()); // Extract width part
        exprt is_signed =
          type == "INT" ? exprt(true_exprt()) : exprt(false_exprt());

        side_effect_expr_function_callt call;
        if (name == "max")
          get_library_function_call_no_args(
            "_max", "c:@F@_max", unsignedbv_typet(256), l, call);
        else
          get_library_function_call_no_args(
            "_min", "c:@F@_min", unsignedbv_typet(256), l, call);
        call.arguments().push_back(constant_exprt(
          integer2binary(string2integer(width), bv_width(uint_type())),
          width,
          uint_type()));
        call.arguments().push_back(is_signed);

        new_expr = call;
        new_expr.location() = l;
        return false;
      }
      else if (name == "creationCode" || name == "runtimeCode")
      {
        // nondet Bytes
        get_library_function_call_no_args(
          "_" + name, "c:@F@_" + name, uint_type(), l, new_expr);
        new_expr.location() = l;
        return false;
      }
      else if (name == "wrap" || name == "unwrap")
      {
        // do nothhing, return operands
        std::string udv = expr["expression"]["name"].get<std::string>();
        assert(UserDefinedVarMap.count(udv) > 0);
        typet t = UserDefinedVarMap[udv];
        new_expr = typecast_exprt(t); // we will set the op0 later
        new_expr.location() = l;
        return false;
      }
      else if (name == "length")
      {
        exprt base;
        if (get_expr(expr["expression"], base))
          return true;
        typet base_t;
        if (get_type_description(
              expr["expression"]["typeDescriptions"], base_t))
          return true;
        std::string solt = base_t.get("#sol_type").as_string();
        if (solt.find("ARRAY") != std::string::npos)
        {
          side_effect_expr_function_callt length_expr;
          get_library_function_call_no_args(
            "_ESBMC_array_length",
            "c:@F@_ESBMC_array_length",
            uint_type(),
            l,
            length_expr);
          length_expr.arguments().push_back(base);

          new_expr = length_expr;
          new_expr.location() = l;
          return false;
        }
        else if (solt.find("BYTES"))
        {
          //TODO
        }
        else
        {
          log_error("Unexpect length of {} type", solt);
          return true;
        }
      }
      else if (name == "push" || name == "pop")
      {
        // _ESBMC_array_push(arr, &val, sizeof(int));
        // push default (0): _ESBMC_array_push(arr, NULL, sizeof(int));
        // _ESBMC_array_pop(arr, sizeof(int));
        exprt base;
        if (get_expr(expr["expression"], base))
          return true;
        typet base_t;
        if (get_type_description(
              expr["expression"]["typeDescriptions"], base_t))
          return true;

        assert(base_t.has_subtype());
        exprt size_of;
        get_size_of_expr(base_t.subtype(), size_of);

        const nlohmann::json &func =
          find_last_parent(src_ast_json["nodes"], expr);
        assert(!func.empty());
        exprt args;
        if (func["arguments"].size() == 0)
          args = gen_zero(pointer_typet(empty_typet()));
        else
        {
          log_status("{}", func.dump());
          if (get_expr(func["arguments"][0], expr["argumentTypes"][0], args))
            return true;
          // wrap it
          std::string aux_name = "_idx#" + std::to_string(aux_counter++);
          std::string aux_id;
          std::string cname;
          get_current_contract_name(expr, cname);
          assert(!cname.empty());
          if (current_functionDecl)
            aux_id =
              "sol:@C@" + cname + "@F@" + current_functionName + "@" + aux_name;
          else
            aux_id = "sol:@C@" + cname + "@" + aux_name;
          symbolt aux_idx;
          get_default_symbol(
            aux_idx,
            get_modulename_from_path(absolute_path),
            args.type(),
            aux_name,
            aux_id,
            l);
          auto &added_aux = *move_symbol_to_context(aux_idx);
          code_declt decl(symbol_expr(added_aux));
          added_aux.value = args;
          decl.operands().push_back(args);
          move_to_front_block(decl);
          args = address_of_exprt(symbol_expr(added_aux));
        }

        side_effect_expr_function_callt mem;
        get_library_function_call_no_args(
          "_ESBMC_array_" + name,
          "c:@F@_ESBMC_array_" + name,
          empty_typet(),
          l,
          mem);

        if (name == "push")
        {
          mem.arguments().push_back(base);
          mem.arguments().push_back(args);
          mem.arguments().push_back(size_of);
        }
        else
        {
          mem.arguments().push_back(base);
          mem.arguments().push_back(size_of);
        }

        new_expr = mem;
        new_expr.location() = l;
        return false;
      }
    }
    if (expr["expression"].contains("name"))
      bs = expr["expression"]["name"].get<std::string>();
    else if (
      expr["expression"].contains("typeName") &&
      expr["expression"]["typeName"].contains("name"))
      bs = expr["expression"]["typeName"]["name"].get<std::string>();
    else
      // cannot get bs name;
      return true;

    std::string mem = expr["memberName"].get<std::string>();
    std::string id_var = "c:@" + bs + "_" + mem;
    std::string id_func = "c:@F@" + bs + "_" + mem;
    if (context.find_symbol(id_var) != nullptr)
    {
      symbolt &sym = *context.find_symbol(id_var);

      if (sym.value.is_empty() || sym.value.is_zero())
      {
        // update: set the value to rand (default 0）
        // since all the current support built-in vars are uint type.
        // we just set the value to c:@F@nondet_uint
        symbolt &r = *context.find_symbol("c:@F@nondet_uint");
        sym.value = r.value;
      }
      new_expr = symbol_expr(sym);
    }

    else if (context.find_symbol(id_func) != nullptr)
      new_expr = symbol_expr(*context.find_symbol(id_func));
    else
      return true;
  }
  else
    return true;

  new_expr.location() = l;
  return false;
}

bool solidity_convertert::get_type_description(
  const nlohmann::json &type_name,
  typet &new_type)
{
  // For Solidity rule type-name:
  SolidityGrammar::TypeNameT type = SolidityGrammar::get_type_name_t(type_name);

  std::string typeIdentifier;
  std::string typeString;

  if (type_name.contains("typeIdentifier"))
    typeIdentifier = type_name["typeIdentifier"].get<std::string>();
  if (type_name.contains("typeString"))
    typeString = type_name["typeString"].get<std::string>();

  log_debug(
    "solidity", "got type-name={}", SolidityGrammar::type_name_to_str(type));

  switch (type)
  {
  case SolidityGrammar::TypeNameT::ElementaryTypeName:
  case SolidityGrammar::TypeNameT::AddressTypeName:
  case SolidityGrammar::TypeNameT::AddressPayableTypeName:
  {
    // rule state-variable-declaration
    if (get_elementary_type_name(type_name, new_type))
      return true;
    break;
  }
  case SolidityGrammar::TypeNameT::ParameterList:
  {
    // rule parameter-list
    // Used for Solidity function parameter or return list
    if (get_parameter_list(type_name, new_type))
      return true;
    break;
  }
  case SolidityGrammar::TypeNameT::Pointer:
  {
    // auxiliary type: pointer (FuncToPtr decay)
    // This part is for FunctionToPointer decay only
    assert(
      typeString.find("function") != std::string::npos ||
      typeString.find("contract") != std::string::npos);

    // Since Solidity does not have this, first make a pointee
    nlohmann::json pointee = make_pointee_type(type_name);
    typet sub_type;
    if (get_func_decl_ref_type(pointee, sub_type))
      return true;

    if (sub_type.is_struct() || sub_type.is_union())
      assert(!"struct or union is NOT supported");

    new_type = gen_pointer_type(sub_type);
    break;
  }
  case SolidityGrammar::TypeNameT::PointerArrayToPtr:
  {
    // auxiliary type: pointer (FuncToPtr decay)
    // This part is for FunctionToPointer decay only
    assert(typeIdentifier.find("ArrayToPtr") != std::string::npos);

    // Array type descriptor is like:
    //  "typeIdentifier": "ArrayToPtr",
    //  "typeString": "uint8[2] memory"

    // Since Solidity does not have this, first make a pointee
    typet sub_type;
    if (get_array_to_pointer_type(type_name, sub_type))
      return true;

    if (
      sub_type.is_struct() ||
      sub_type.is_union()) // for "assert(sum > 100)", false || false
      assert(!"struct or union is NOT supported");

    new_type = gen_pointer_type(sub_type);
    break;
  }
  case SolidityGrammar::TypeNameT::ArrayTypeName:
  case SolidityGrammar::TypeNameT::DynArrayTypeName:
  {
    // Deal with array with constant size, e.g., int a[2]; Similar to clang::Type::ConstantArray
    // array's typeDescription is in a compact form, e.g.:
    //    "typeIdentifier": "t_array$_t_uint8_$2_storage_ptr",
    //    "typeString": "uint8[2]"
    // We need to extract the elementary type of array from the information provided above
    // We want to make it like ["baseType"]["typeDescriptions"]

    typet the_type;
    exprt the_size;
    if (current_typeName != nullptr)
    {
      // access from get_var_decl
      assert((*current_typeName).contains("baseType"));
      if (get_type_description(
            (*current_typeName)["baseType"]["typeDescriptions"], the_type))
        return true;

      new_type = gen_pointer_type(the_type);
      if ((*current_typeName).contains("length"))
      {
        assert(type == SolidityGrammar::TypeNameT::ArrayTypeName);
        std::string length =
          (*current_typeName)["length"]["value"].get<std::string>();
        new_type.set("#sol_array_size", length);
        new_type.set("#sol_type", "ARRAY");
      }
      else
      {
        assert(type == SolidityGrammar::TypeNameT::DynArrayTypeName);
        new_type.set("#sol_type", "DYNARRAY");
      }
    }
    else if (type == SolidityGrammar::TypeNameT::ArrayTypeName)
    {
      // for tuple array
      nlohmann::json array_elementary_type =
        make_array_elementary_type(type_name);

      if (get_type_description(array_elementary_type, the_type))
        return true;

      assert(the_type.is_unsignedbv()); // assuming array size is unsigned bv
      std::string the_size = get_array_size(type_name);
      unsigned z_ext_value = std::stoul(the_size, nullptr);
      new_type = array_typet(
        the_type,
        constant_exprt(
          integer2binary(z_ext_value, bv_width(int_type())),
          integer2string(z_ext_value),
          int_type()));
      new_type.set("#sol_array_size", the_size);
      new_type.set("#sol_type", "ARRAY_LITERAL");
    }
    else
    {
      // e.g.
      // "typeDescriptions": {
      //     "typeIdentifier": "t_array$_t_uint256_$dyn_memory_ptr",
      //     "typeString": "uint256[]"

      // 1. rebuild baseType
      nlohmann::json new_json;
      std::string temp = typeString;
      auto pos = temp.find("[]"); // e.g. "uint256[] memory"
      const std::string new_typeString = temp.substr(0, pos);
      const std::string new_typeIdentifier = "t_" + new_typeString;
      new_json["typeString"] = new_typeString;
      new_json["typeIdentifier"] = new_typeIdentifier;

      // 2. get subType
      typet sub_type;
      if (get_type_description(new_json, sub_type))
        return true;

      // 3. make pointer
      new_type = gen_pointer_type(sub_type);
      new_type.set("#sol_type", "DYNARRAY");
    }

    break;
  }
  case SolidityGrammar::TypeNameT::ContractTypeName:
  {
    // e.g. ContractName tmp = new ContractName(Args);

    std::string constructor_name = typeString;
    size_t pos = constructor_name.find(" ");
    std::string cname = constructor_name.substr(pos + 1);
    std::string id = prefix + cname;

    new_type = pointer_typet(symbol_typet(id));
    new_type.set("#sol_type", "CONTRACT");
    new_type.set("#sol_contract", cname);
    break;
  }
  case SolidityGrammar::TypeNameT::TypeConversionName:
  {
    // e.g.
    // uint32 a = 0x432178;
    // uint16 b = uint16(a); // b will be 0x2178 now
    // "nodeType": "TypeConversionExpression",
    //             "src": "155:6:0",
    //             "typeDescriptions": {
    //                 "typeIdentifier": "t_type$_t_uint16_$",
    //                 "typeString": "type(uint16)"
    //             },
    //             "typeName": {
    //                 "id": 10,
    //                 "name": "uint16",
    //                 "nodeType": "ElementaryTypeName",
    //                 "src": "155:6:0",
    //                 "typeDescriptions": {}
    //             }

    nlohmann::json new_json;

    // convert it back to ElementaryTypeName by removing the "type" prefix
    std::size_t begin = typeIdentifier.find("$_");
    std::size_t end = typeIdentifier.rfind("_$");
    std::string new_typeIdentifier =
      typeIdentifier.substr(begin + 2, end - begin - 2);

    begin = typeString.find("type(");
    end = typeString.rfind(")");
    std::string new_typeString = typeString.substr(begin + 5, end - begin - 5);

    new_json["typeIdentifier"] = new_typeIdentifier;
    new_json["typeString"] = new_typeString;

    if (get_type_description(new_json, new_type))
      return true;

    break;
  }
  case SolidityGrammar::TypeNameT::EnumTypeName:
  {
    new_type = enum_type();
    new_type.set("#sol_type", "ENUM");
    break;
  }
  case SolidityGrammar::TypeNameT::StructTypeName:
  {
    // e.g. struct ContractName.StructName
    //   "typeDescriptions": {
    //   "typeIdentifier": "t_struct$_Book_$8_storage",
    //   "typeString": "struct Base.Book storage ref"
    // }

    // extract id and ref_id;
    std::string delimiter = " ";

    int cnt = 1;
    std::string token;
    std::string _typeString = typeString;

    // extract the second string
    while (cnt >= 0)
    {
      if (_typeString.find(delimiter) == std::string::npos)
      {
        token = _typeString;
        break;
      }
      size_t pos = _typeString.find(delimiter);
      token = _typeString.substr(0, pos);
      _typeString.erase(0, pos + delimiter.length());
      cnt--;
    }

    const std::string id = prefix + "struct " + token;
    new_type = symbol_typet(id);
    new_type.set("#sol_type", "STRUCT");
    break;
  }
  case SolidityGrammar::TypeNameT::MappingTypeName:
  {
    /*
        "typeIdentifier": "t_mapping$_t_address_$_t_uint256_$",
        "typeString": "mapping(address => uint256)"
    */
    // we need to check if it's inside a contract used in a new expression statement
    assert(!current_baseContractName.empty());
    bool is_new_expr = newContractSet.count(current_baseContractName);
    if (is_new_expr)
    {
      if (
        !is_bound && tgt_cnt_set.count(current_baseContractName) > 0 &&
        tgt_cnt_set.size() == 1)
        is_new_expr = false;
    }

    if (is_new_expr)
      new_type = symbol_typet(prefix + "mapping_t");
    else
    {
      // we will populate the size type later
      new_type = array_typet();
      new_type.size(exprt("infinity"));
    }
    new_type.set("#sol_type", "MAPPING");
    break;
  }
  case SolidityGrammar::TypeNameT::TupleTypeName:
  {
    // do nothing as it won't be used
    new_type = struct_typet();
    new_type.set("#cpp_type", "void");
    new_type.set("#sol_type", "TUPLE_RETURNS");
    break;
  }
  case SolidityGrammar::TypeNameT::UserDefinedTypeName:
  {
    new_type = UserDefinedVarMap[typeString];
    break;
  }
  default:
  {
    log_debug(
      "solidity",
      "	@@@ got type name=SolidityGrammar::TypeNameT::{}",
      SolidityGrammar::type_name_to_str(type));
    assert(!"Unimplemented type in rule type-name");
    return true;
  }
  }

  // TODO: More var decl attributes checks:
  //    - Constant
  //    - Volatile
  //    - isRestrict

  // set data location
  if (typeIdentifier.find("_memory_ptr") != std::string::npos)
    new_type.set("#sol_data_loc", "memory");
  else if (typeIdentifier.find("_storage_ptr") != std::string::npos)
    new_type.set("#sol_data_loc", "storage");
  else if (typeIdentifier.find("_calldata_ptr") != std::string::npos)
    new_type.set("#sol_data_loc", "calldata");

  return false;
}

bool solidity_convertert::get_func_decl_ref_type(
  const nlohmann::json &decl,
  typet &new_type)
{
  // For FunctionToPointer decay:
  // Get type when we make a function call:
  //  - FunnctionNoProto: x = nondet()
  //  - FunctionProto:    z = add(x, y)
  // Similar to the function get_type_description()
  SolidityGrammar::FunctionDeclRefT type =
    SolidityGrammar::get_func_decl_ref_t(decl);

  log_debug(
    "solidity",
    "\t@@@ got SolidityGrammar::FunctionDeclRefT = {}",
    SolidityGrammar::func_decl_ref_to_str(type));

  switch (type)
  {
  case SolidityGrammar::FunctionDeclRefT::FunctionNoProto:
  {
    code_typet type;
    // Return type
    const nlohmann::json &rtn_type = decl["returnParameters"];

    typet return_type;
    if (get_type_description(rtn_type, return_type))
      return true;

    type.return_type() = return_type;

    if (!type.arguments().size())
      type.make_ellipsis();

    new_type = type;
    break;
  }
  case SolidityGrammar::FunctionDeclRefT::FunctionProto:
  {
    code_typet type;

    // store current state
    const nlohmann::json *old_functionDecl = current_functionDecl;
    const std::string old_functionName = current_functionName;

    // need in get_function_params()
    assert(decl.contains("name"));
    current_functionName = decl["name"].get<std::string>();
    current_functionDecl = &decl;

    std::string current_contractName;
    get_current_contract_name(decl, current_contractName);

    if (decl.contains("returnParameters"))
    {
      const nlohmann::json &rtn_type = decl["returnParameters"];

      typet return_type;
      if (get_type_description(rtn_type, return_type))
        return true;

      type.return_type() = return_type;
    }

    // convert parameters if the function has them
    // update the typet, since typet contains parameter annotations
    for (const auto &decl : decl["parameters"]["parameters"].items())
    {
      const nlohmann::json &func_param_decl = decl.value();

      code_typet::argumentt param;
      if (get_function_params(func_param_decl, current_contractName, param))
        return true;

      type.arguments().push_back(param);
    }

    current_functionName = old_functionName;
    current_functionDecl = old_functionDecl;

    new_type = type;
    break;
  }
  default:
  {
    log_debug(
      "solidity",
      "	@@@ Got type={}",
      SolidityGrammar::func_decl_ref_to_str(type));
    //TODO: seem to be unnecessary, need investigate
    // assert(!"Unimplemented type in auxiliary type to convert function call");
    return true;
  }
  }

  // TODO: More var decl attributes checks:
  //    - Constant
  //    - Volatile
  //    - isRestrict
  return false;
}

bool solidity_convertert::get_array_to_pointer_type(
  const nlohmann::json &type_descriptor,
  typet &new_type)
{
  // Function to get the base type in ArrayToPointer decay
  //  - unrolled the get_type...
  if (
    type_descriptor["typeString"].get<std::string>().find("uint8") !=
    std::string::npos)
  {
    new_type = unsigned_char_type();
    new_type.set("#cpp_type", "unsigned_char");
  }
  else
    assert(!"Unsupported types in ArrayToPinter decay");

  // TODO: More var decl attributes checks:
  //    - Constant
  //    - Volatile
  //    - isRestrict
  return false;
}

// parse a tuple to struct
bool solidity_convertert::get_tuple_definition(const nlohmann::json &ast_node)
{
  log_debug("solidity", "\t@@@ Parsing tuple...");

  std::string current_contractName;
  get_current_contract_name(ast_node, current_contractName);
  if (current_contractName.empty())
  {
    log_error(
      "Cannot get the contract name. Tuple should always within a contract.");
    return true;
  }

  struct_typet t = struct_typet();

  // get name/id:
  std::string name, id;
  get_tuple_name(ast_node, name, id);

  // get type:
  t.tag("struct " + name);

  // get location
  locationt location_begin;
  get_location_from_node(ast_node, location_begin);

  // get debug module name
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  // populate struct type symbol
  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, t, name, id, location_begin);
  symbol.static_lifetime = true;
  symbol.file_local = true;
  symbolt &added_symbol = *move_symbol_to_context(symbol);

  auto &args = ast_node.contains("components")
                 ? ast_node["components"]
                 : ast_node["returnParameters"]["parameters"];

  // populate params
  //TODO: flatten the nested tuple (e.g. ((x,y),z) = (func(),1); )
  size_t counter = 0;
  for (const auto &arg : args.items())
  {
    if (arg.value().is_null())
    {
      ++counter;
      continue;
    }

    struct_typet::componentt comp;

    // manually create a member_name
    // follow the naming rule defined in get_local_var_decl_name
    assert(!current_contractName.empty());
    const std::string mem_name = "mem" + std::to_string(counter);
    const std::string mem_id = "sol:@C@" + current_contractName + "@" + name +
                               "@" + mem_name + "#" +
                               i2string(ast_node["id"].get<std::int16_t>());

    // get type
    typet mem_type;
    if (get_type_description(arg.value()["typeDescriptions"], mem_type))
      return true;

    // construct comp
    comp.type() = mem_type;
    comp.type().set("#member_name", t.tag());
    comp.identifier(mem_id);
    comp.cmt_lvalue(true);
    comp.name(mem_name);
    comp.pretty_name(mem_name);
    comp.set_access("internal");

    // update struct type component
    t.components().push_back(comp);

    // update cnt
    ++counter;
  }

  t.location() = location_begin;
  added_symbol.type = t;

  return false;
}

bool solidity_convertert::get_tuple_instance(
  const nlohmann::json &ast_node,
  exprt &new_expr)
{
  std::string name, id;
  get_tuple_name(ast_node, name, id);

  if (context.find_symbol(id) == nullptr)
    return true;

  // get type
  typet t = context.find_symbol(id)->type;
  t.set("#sol_type", "TUPLE_INSTANCE");
  assert(t.id() == typet::id_struct);

  // get instance name,id
  if (get_tuple_instance_name(ast_node, name, id))
    return true;

  // get location
  locationt location_begin;
  get_location_from_node(ast_node, location_begin);

  // get debug module name
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  // populate struct type symbol
  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, t, name, id, location_begin);
  symbol.static_lifetime = true;
  symbol.file_local = true;

  symbol.value = gen_zero(get_complete_type(t, ns), true);
  symbol.value.zero_initializer(true);
  symbolt &added_symbol = *move_symbol_to_context(symbol);
  new_expr = symbol_expr(added_symbol);
  new_expr.identifier(id);

  if (!ast_node.contains("components"))
  {
    // assume it's function return parameter list
    // therefore no initial value
    return false;
  }

  // do assignment
  auto &args = ast_node["components"];

  size_t i = 0;
  size_t j = 0;
  unsigned is = to_struct_type(t).components().size();
  unsigned as = args.size();
  assert(is <= as);

  exprt comp;
  exprt member_access;
  while (i < is && j < as)
  {
    if (args.at(j).is_null())
    {
      ++j;
      continue;
    }

    comp = to_struct_type(t).components().at(i);
    if (get_tuple_member_call(id, comp, member_access))
      return true;

    exprt init;
    const nlohmann::json &litera_type = args.at(j)["typeDescriptions"];

    if (get_expr(args.at(j), litera_type, init))
      return true;

    get_tuple_assignment(member_access, init);

    // update
    ++i;
    ++j;
  }

  return false;
}

void solidity_convertert::get_tuple_name(
  const nlohmann::json &ast_node,
  std::string &name,
  std::string &id)
{
  name = "tuple" + std::to_string(ast_node["id"].get<int>());
  id = prefix + "struct " + name;
}

bool solidity_convertert::get_tuple_instance_name(
  const nlohmann::json &ast_node,
  std::string &name,
  std::string &id)
{
  std::string c_name;
  get_current_contract_name(ast_node, c_name);
  if (c_name.empty())
    return true;

  name = "tuple_instance$" + std::to_string(ast_node["id"].get<int>());
  id = "sol:@C@" + c_name + "@" + name;
  return false;
}

/*
  obtain the corresponding tuple struct instance from the symbol table 
  based on the function definition json
*/
bool solidity_convertert::get_tuple_function_ref(
  const nlohmann::json &ast_node,
  exprt &new_expr)
{
  assert(ast_node.contains("nodeType") && ast_node["nodeType"] == "Identifier");

  std::string c_name;
  get_current_contract_name(ast_node, c_name);
  if (c_name.empty())
    return true;

  std::string name =
    "tuple_instance$" +
    std::to_string(ast_node["referencedDeclaration"].get<int>());
  std::string id = "sol:@C@" + c_name + "@" + name;

  if (context.find_symbol(id) == nullptr)
    return true;

  new_expr = symbol_expr(*context.find_symbol(id));
  return false;
}

// Knowing that there is a component x in the struct_tuple_instance A, we construct A.x
bool solidity_convertert::get_tuple_member_call(
  const irep_idt instance_id,
  const exprt &comp,
  exprt &new_expr)
{
  // tuple_instance
  assert(!instance_id.empty());
  exprt base;
  if (context.find_symbol(instance_id) == nullptr)
    return true;

  base = symbol_expr(*context.find_symbol(instance_id));
  new_expr = member_exprt(base, comp.name(), comp.type());
  return false;
}

void solidity_convertert::get_tuple_function_call(const exprt &op)
{
  assert(op.id() == "sideeffect");
  exprt func_call = op;
  convert_expression_to_code(func_call);
  if (current_functionDecl)
    move_to_back_block(func_call);
  else
    move_to_initializer(func_call);
}

void solidity_convertert::get_llc_ret_tuple(symbolt &s)
{
  log_debug("solidity", "\tconvert return value to tuple");
  std::string _id = "tag-sol_llc_ret";
  if (context.find_symbol(_id) == nullptr)
  {
    log_error("cannot find library symbol tag-sol_llc_ret");
    abort();
  }
  const symbolt &struct_sym = *context.find_symbol(_id);

  typet sym_t = struct_sym.type;
  sym_t.set("#sol_type", "TUPLE_INSTANCE");

  std::string name, id;
  name = "tuple_instance$" + std::to_string(aux_counter);
  id = "sol:@" + name;
  locationt l;
  symbolt symbol;
  get_default_symbol(
    symbol, get_modulename_from_path(absolute_path), sym_t, name, id, l);
  symbol.static_lifetime = true;
  symbol.file_local = true;
  auto &added_sym = *move_symbol_to_context(symbol);

  // value
  typet t = struct_sym.type;
  exprt inits = gen_zero(t);
  inits.op0() = nondet_bool_expr;
  inits.op1() = nondet_uint_expr;
  added_sym.value = inits;
  s = added_sym;
}

void solidity_convertert::get_string_assignment(
  const exprt &lhs,
  const exprt &rhs,
  exprt &new_expr)
{
  side_effect_expr_function_callt call;
  get_str_assign_function_call(lhs.location(), call);
  call.arguments().push_back(address_of_exprt(lhs));
  call.arguments().push_back(rhs);
  new_expr = call;
}

/*
  lhs: code_blockt
  rhs: tuple_return / tuple_instancce
*/
bool solidity_convertert::construct_tuple_assigments(
  const nlohmann::json &expr,
  const exprt &lhs,
  const exprt &rhs)
{
  log_debug("solidity", "Handling tuple assignment.");

  typet lt = lhs.type();
  typet rt = rhs.type();
  std::string lt_sol = lt.get("#sol_type").as_string();
  std::string rt_sol = rt.get("#sol_type").as_string();

  assert(lt.is_code() && to_code(lhs).statement() == "block");
  exprt new_rhs = rhs;
  if (rt_sol == "TUPLE_RETURNS")
  {
    // e.g. (x,y) = func(); (x,y) = func(func2()); (x, (x,y)) = (x, func());
    // ==>
    //    func(); // this initializes the tuple instance
    //    x = tuple.mem0;
    //    y = tuple.mem1;
    if (get_tuple_function_ref(expr["rightHandSide"]["expression"], new_rhs))
      return true;

    // add function call
    get_tuple_function_call(rhs);
  }

  // e.g. (x,y) = (1,2); (x,y) = (func(),x);
  // =>
  //  t.mem0 = 1; #1
  //  t.mem1 = 2; #2
  //  x = t.mem0; #3
  //  y = t.mem1; #4
  // where #1 and #2 are already in the expr_backBlockDecl

  // do #3 #4
  std::set<exprt> assigned_symbol;
  for (size_t i = 0; i < lhs.operands().size(); i++)
  {
    // e.g. (, x) = (1, 2)
    //      null <=> tuple2.mem0
    //         x <=> tuple2.mem1
    exprt lop = lhs.operands().at(i);
    if (lop.is_nil() || assigned_symbol.count(lop))
      // e.g. (,y) = (1,2)
      // or   (x,x) = (1, 2); assert(x==1) hold
      // we skip the variable that has been assigned
      continue;
    assigned_symbol.insert(lop);

    exprt rop;
    if (!new_rhs.type().is_struct())
    {
      log_error("expecting struct type, got {}", new_rhs);
      return true;
    }
    if (get_tuple_member_call(
          new_rhs.identifier(),
          to_struct_type(new_rhs.type()).components().at(i),
          rop))
      return true;

    get_tuple_assignment(lop, rop);
  }
  return false;
}

void solidity_convertert::get_tuple_assignment(const exprt &lop, exprt rop)
{
  exprt assign_expr;
  if (lop.type().get("#sol_type") == "STRING")
    get_string_assignment(lop, rop, assign_expr);
  else
  {
    assign_expr = side_effect_exprt("assign", lop.type());
    convert_type_expr(ns, rop, lop.type());
    assign_expr.copy_to_operands(lop, rop);
  }
  convert_expression_to_code(assign_expr);
  if (current_functionDecl)
    move_to_back_block(assign_expr);
  else
    move_to_initializer(assign_expr);
}

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
  std::string &key_sol_type,
  std::string &val_sol_type)
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
  key_sol_type = key_t.get("#sol_type").as_string();
  val_sol_type = value_t.get("#sol_type").as_string();
  if (val_sol_type.empty())
    return true;
  return false;
}

void solidity_convertert::gen_mapping_key_typecast(
  exprt &pos,
  const locationt &location,
  const std::string &key_sol_type)
{
  // if index is a string, we should convert it to uint256
  if (key_sol_type == "STRING" || key_sol_type == "STRING_LITERAL")
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
  }
  // e.g x[-1] => x[uint256(-1)]
  solidity_gen_typecast(ns, pos, unsignedbv_typet(256));
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
  const std::string &val_sol_type,
  bool is_mapping_set,
  const exprt &array,
  const exprt &pos,
  const locationt &location,
  exprt &new_expr)
{
  std::string val_flg;
  typet func_type;
  if (
    val_sol_type.find("UINT") != std::string::npos ||
    val_sol_type.find("BYTES") != std::string::npos ||
    val_sol_type.find("ADDRESS") != std::string::npos ||
    val_sol_type.find("ENUM") != std::string::npos)
  {
    val_flg = "uint";
    func_type = unsignedbv_typet(256);
  }
  else if (val_sol_type.find("INT") != std::string::npos)
  {
    val_flg = "int";
    func_type = signedbv_typet(256);
  }
  else if (val_sol_type.find("BOOL") != std::string::npos)
  {
    val_flg = "bool";
    func_type = bool_typet();
  }
  else if (
    val_sol_type.find("STRING") != std::string::npos ||
    val_sol_type.find("STRING_LITERAL") != std::string::npos)
  {
    val_flg = "string";
    func_type = value_t;
  }
  else
  {
    val_flg = "generic";
    // void *
    func_type = pointer_typet(empty_typet());
  }

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
      val_sol_type);
    return true;
  }
  side_effect_expr_function_callt call;
  get_library_function_call_no_args(
    func_name, "c:@F@" + func_name, func_type, location, call);

  // &array
  call.arguments().push_back(address_of_exprt(array));

  // index
  call.arguments().push_back(pos);

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
      call.arguments().push_back(symbol_expr(added_sym));
    }

    convert_expression_to_code(call);
    move_to_back_block(call);

    new_expr = symbol_expr(added_sym);
  }
  else if (val_flg == "generic")
  {
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
    assert(!struct_contract_name.empty());
    assert(val_sol_type == "STRUCT"); // t_symbol
    get_mapping_struct_function(
      value_t, struct_contract_name, call, map_struct_get);

    // struct temp = map_users_get(&array, pos);
    added_sym.value = map_struct_get;
    decl.operands().push_back(map_struct_get);
    move_to_front_block(decl);

    new_expr = symbol_expr(added_sym);
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
  // for typcast
  side_effect_expr_function_callt temp_call = gen_call;
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
  const std::string calc_name = "calloc";
  const std::string calc_id = "c:@F@calloc";
  const symbolt &calc_sym = *context.find_symbol(calc_id);
  get_library_function_call_no_args(
    calc_name, calc_id, symbol_expr(calc_sym).type(), loc, calc_call);
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

/**
     * @brief Populate the out `typet` parameter with the uint type specified by type parameter
     *
     * @param type The type of the uint to be poulated
     * @param out The variable that holds the resulting type
     * @return true iff population failed
     * @return false iff population was successful
     */
bool solidity_convertert::get_elementary_type_name_uint(
  SolidityGrammar::ElementaryTypeNameT &type,
  typet &out)
{
  const unsigned int uint_size = SolidityGrammar::uint_type_name_to_size(type);
  out = unsignedbv_typet(uint_size);

  return false;
}

/**
     * @brief Populate the out `typet` parameter with the int type specified by type parameter
     *
     * @param type The type of the int to be poulated
     * @param out The variable that holds the resulting type
     * @return false iff population was successful
     */
bool solidity_convertert::get_elementary_type_name_int(
  SolidityGrammar::ElementaryTypeNameT &type,
  typet &out)
{
  const unsigned int int_size = SolidityGrammar::int_type_name_to_size(type);
  out = signedbv_typet(int_size);

  return false;
}

bool solidity_convertert::get_elementary_type_name_bytesn(
  SolidityGrammar::ElementaryTypeNameT &type,
  typet &out)
{
  /*
    bytes1 has size of 8 bits (possible values 0x00 to 0xff), 
    which you can implicitly convert to uint8 (unsigned integer of size 8 bits) but not to int8
  */
  const unsigned int byte_num = SolidityGrammar::bytesn_type_name_to_size(type);
  out = unsignedbv_typet(byte_num * 8);

  return false;
}

bool solidity_convertert::get_elementary_type_name(
  const nlohmann::json &type_name,
  typet &new_type)
{
  // For Solidity rule elementary-type-name:
  // equivalent to clang's get_builtin_type()
  std::string c_type;
  SolidityGrammar::ElementaryTypeNameT type =
    SolidityGrammar::get_elementary_type_name_t(type_name);

  log_debug(
    "solidity",
    "	@@@ got ElementaryType: SolidityGrammar::ElementaryTypeNameT::{}",
    fmt::underlying(type));

  switch (type)
  {
  // rule unsigned-integer-type
  case SolidityGrammar::ElementaryTypeNameT::UINT8:
  case SolidityGrammar::ElementaryTypeNameT::UINT16:
  case SolidityGrammar::ElementaryTypeNameT::UINT24:
  case SolidityGrammar::ElementaryTypeNameT::UINT32:
  case SolidityGrammar::ElementaryTypeNameT::UINT40:
  case SolidityGrammar::ElementaryTypeNameT::UINT48:
  case SolidityGrammar::ElementaryTypeNameT::UINT56:
  case SolidityGrammar::ElementaryTypeNameT::UINT64:
  case SolidityGrammar::ElementaryTypeNameT::UINT72:
  case SolidityGrammar::ElementaryTypeNameT::UINT80:
  case SolidityGrammar::ElementaryTypeNameT::UINT88:
  case SolidityGrammar::ElementaryTypeNameT::UINT96:
  case SolidityGrammar::ElementaryTypeNameT::UINT104:
  case SolidityGrammar::ElementaryTypeNameT::UINT112:
  case SolidityGrammar::ElementaryTypeNameT::UINT120:
  case SolidityGrammar::ElementaryTypeNameT::UINT128:
  case SolidityGrammar::ElementaryTypeNameT::UINT136:
  case SolidityGrammar::ElementaryTypeNameT::UINT144:
  case SolidityGrammar::ElementaryTypeNameT::UINT152:
  case SolidityGrammar::ElementaryTypeNameT::UINT160:
  case SolidityGrammar::ElementaryTypeNameT::UINT168:
  case SolidityGrammar::ElementaryTypeNameT::UINT176:
  case SolidityGrammar::ElementaryTypeNameT::UINT184:
  case SolidityGrammar::ElementaryTypeNameT::UINT192:
  case SolidityGrammar::ElementaryTypeNameT::UINT200:
  case SolidityGrammar::ElementaryTypeNameT::UINT208:
  case SolidityGrammar::ElementaryTypeNameT::UINT216:
  case SolidityGrammar::ElementaryTypeNameT::UINT224:
  case SolidityGrammar::ElementaryTypeNameT::UINT232:
  case SolidityGrammar::ElementaryTypeNameT::UINT240:
  case SolidityGrammar::ElementaryTypeNameT::UINT248:
  case SolidityGrammar::ElementaryTypeNameT::UINT256:
  {
    if (get_elementary_type_name_uint(type, new_type))
      return true;

    new_type.set("#sol_type", elementary_type_name_to_str(type));
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::INT8:
  case SolidityGrammar::ElementaryTypeNameT::INT16:
  case SolidityGrammar::ElementaryTypeNameT::INT24:
  case SolidityGrammar::ElementaryTypeNameT::INT32:
  case SolidityGrammar::ElementaryTypeNameT::INT40:
  case SolidityGrammar::ElementaryTypeNameT::INT48:
  case SolidityGrammar::ElementaryTypeNameT::INT56:
  case SolidityGrammar::ElementaryTypeNameT::INT64:
  case SolidityGrammar::ElementaryTypeNameT::INT72:
  case SolidityGrammar::ElementaryTypeNameT::INT80:
  case SolidityGrammar::ElementaryTypeNameT::INT88:
  case SolidityGrammar::ElementaryTypeNameT::INT96:
  case SolidityGrammar::ElementaryTypeNameT::INT104:
  case SolidityGrammar::ElementaryTypeNameT::INT112:
  case SolidityGrammar::ElementaryTypeNameT::INT120:
  case SolidityGrammar::ElementaryTypeNameT::INT128:
  case SolidityGrammar::ElementaryTypeNameT::INT136:
  case SolidityGrammar::ElementaryTypeNameT::INT144:
  case SolidityGrammar::ElementaryTypeNameT::INT152:
  case SolidityGrammar::ElementaryTypeNameT::INT160:
  case SolidityGrammar::ElementaryTypeNameT::INT168:
  case SolidityGrammar::ElementaryTypeNameT::INT176:
  case SolidityGrammar::ElementaryTypeNameT::INT184:
  case SolidityGrammar::ElementaryTypeNameT::INT192:
  case SolidityGrammar::ElementaryTypeNameT::INT200:
  case SolidityGrammar::ElementaryTypeNameT::INT208:
  case SolidityGrammar::ElementaryTypeNameT::INT216:
  case SolidityGrammar::ElementaryTypeNameT::INT224:
  case SolidityGrammar::ElementaryTypeNameT::INT232:
  case SolidityGrammar::ElementaryTypeNameT::INT240:
  case SolidityGrammar::ElementaryTypeNameT::INT248:
  case SolidityGrammar::ElementaryTypeNameT::INT256:
  {
    if (get_elementary_type_name_int(type, new_type))
      return true;

    new_type.set("#sol_type", elementary_type_name_to_str(type));
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::INT_LITERAL:
  {
    // for int_const type
    new_type = signedbv_typet(256);
    new_type.set("#cpp_type", "signed_char");
    new_type.set("#sol_type", "INT_CONST");
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::BOOL:
  {
    new_type = bool_type();
    new_type.set("#cpp_type", "bool");
    new_type.set("#sol_type", "BOOL");
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::STRING:
  {
    // cpp: std::string str;
    // new_type = symbol_typet("tag-std::string");
    new_type = pointer_typet(signed_char_type());
    new_type.set("#sol_type", "STRING");
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::ADDRESS:
  case SolidityGrammar::ElementaryTypeNameT::ADDRESS_PAYABLE:
  {
    //  An Address is a DataHexString of 20 bytes (uint160)
    // e.g. 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984
    // ops: <=, <, ==, !=, >= and >

    new_type = unsignedbv_typet(160);

    // for type conversion
    new_type.set("#sol_type", elementary_type_name_to_str(type));
    new_type.set("#sol_type", "ADDRESS");
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::BYTES1:
  case SolidityGrammar::ElementaryTypeNameT::BYTES2:
  case SolidityGrammar::ElementaryTypeNameT::BYTES3:
  case SolidityGrammar::ElementaryTypeNameT::BYTES4:
  case SolidityGrammar::ElementaryTypeNameT::BYTES5:
  case SolidityGrammar::ElementaryTypeNameT::BYTES6:
  case SolidityGrammar::ElementaryTypeNameT::BYTES7:
  case SolidityGrammar::ElementaryTypeNameT::BYTES8:
  case SolidityGrammar::ElementaryTypeNameT::BYTES9:
  case SolidityGrammar::ElementaryTypeNameT::BYTES10:
  case SolidityGrammar::ElementaryTypeNameT::BYTES11:
  case SolidityGrammar::ElementaryTypeNameT::BYTES12:
  case SolidityGrammar::ElementaryTypeNameT::BYTES13:
  case SolidityGrammar::ElementaryTypeNameT::BYTES14:
  case SolidityGrammar::ElementaryTypeNameT::BYTES15:
  case SolidityGrammar::ElementaryTypeNameT::BYTES16:
  case SolidityGrammar::ElementaryTypeNameT::BYTES17:
  case SolidityGrammar::ElementaryTypeNameT::BYTES18:
  case SolidityGrammar::ElementaryTypeNameT::BYTES19:
  case SolidityGrammar::ElementaryTypeNameT::BYTES20:
  case SolidityGrammar::ElementaryTypeNameT::BYTES21:
  case SolidityGrammar::ElementaryTypeNameT::BYTES22:
  case SolidityGrammar::ElementaryTypeNameT::BYTES23:
  case SolidityGrammar::ElementaryTypeNameT::BYTES24:
  case SolidityGrammar::ElementaryTypeNameT::BYTES25:
  case SolidityGrammar::ElementaryTypeNameT::BYTES26:
  case SolidityGrammar::ElementaryTypeNameT::BYTES27:
  case SolidityGrammar::ElementaryTypeNameT::BYTES28:
  case SolidityGrammar::ElementaryTypeNameT::BYTES29:
  case SolidityGrammar::ElementaryTypeNameT::BYTES30:
  case SolidityGrammar::ElementaryTypeNameT::BYTES31:
  case SolidityGrammar::ElementaryTypeNameT::BYTES32:
  {
    if (get_elementary_type_name_bytesn(type, new_type))
      return true;

    // for type conversion
    new_type.set("#sol_type", elementary_type_name_to_str(type));
    new_type.set("#sol_bytes_size", bytesn_type_name_to_size(type));

    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::BYTES:
  {
    new_type = unsignedbv_typet(256);
    new_type.set("#sol_type", elementary_type_name_to_str(type));

    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::STRING_LITERAL:
  {
    // it's fine even if the json is not the exact parent
    auto json = find_last_parent(src_ast_json, type_name);
    assert(!json.empty());
    string_constantt x(json["value"].get<std::string>());
    new_type = x.type();
    new_type.set("#sol_type", "STRING_LITERAL");

    break;
  }
  default:
  {
    log_debug(
      "solidity",
      "	@@@ Got elementary-type-name={}",
      SolidityGrammar::elementary_type_name_to_str(type));
    assert(!"Unimplemented type in rule elementary-type-name");
    return true;
  }
  }

  //TODO set #extint
  // switch (type)
  // {
  // case SolidityGrammar::ElementaryTypeNameT::BOOL:
  // case SolidityGrammar::ElementaryTypeNameT::STRING:
  // {
  //   break;
  // }
  // default:
  // {
  //   new_type.set("#extint", true);
  //   break;
  // }
  // }

  return false;
}

bool solidity_convertert::get_parameter_list(
  const nlohmann::json &type_name,
  typet &new_type)
{
  // For Solidity rule parameter-list:
  //  - For non-empty param list, it may need to call get_elementary_type_name, since parameter-list is just a list of types
  std::string c_type;
  SolidityGrammar::ParameterListT type =
    SolidityGrammar::get_parameter_list_t(type_name);

  log_debug(
    "solidity",
    "\tGot ParameterList {}",
    SolidityGrammar::parameter_list_to_str(type));

  switch (type)
  {
  case SolidityGrammar::ParameterListT::EMPTY:
  {
    // equivalent to clang's "void"
    new_type = empty_typet();
    c_type = "void";
    new_type.set("#cpp_type", c_type);
    break;
  }
  case SolidityGrammar::ParameterListT::ONE_PARAM:
  {
    assert(type_name["parameters"].size() == 1);

    const nlohmann::json &rtn_type = type_name["parameters"].at(0);
    if (rtn_type.contains("typeName"))
    {
      const nlohmann::json *old_typeName = current_typeName;
      current_typeName = &rtn_type["typeName"];
      if (get_type_description(
            rtn_type["typeName"]["typeDescriptions"], new_type))
        return true;
      current_typeName = old_typeName;
    }
    else
    {
      if (get_type_description(rtn_type["typeDescriptions"], new_type))
        return true;
    }

    break;
  }
  case SolidityGrammar::ParameterListT::MORE_THAN_ONE_PARAM:
  {
    // if contains multiple return types
    // We will return null because we create the symbols of the struct accordingly
    assert(type_name["parameters"].size() > 1);
    new_type = empty_typet();
    new_type.set("#cpp_type", "void");
    new_type.set("#sol_type", "TUPLE_RETURNS");
    break;
  }
  default:
  {
    assert(!"Unimplemented type in rule parameter-list");
    return true;
  }
  }

  return false;
}

// parse the state variable
void solidity_convertert::get_state_var_decl_name(
  const nlohmann::json &ast_node,
  const std::string &cname,
  std::string &name,
  std::string &id)
{
  // Follow the way in clang:
  //  - For state variable name, just use the ast_node["name"], e.g. sol:@C@Base@x#11
  //  - For state variable id, add prefix "sol:@"
  name = ast_node["name"].get<std::string>();
  if (!cname.empty())
    id = "sol:@C@" + cname + "@" + name + "#" +
         i2string(ast_node["id"].get<std::int16_t>());
  else
    id = "sol:@" + name + "#" + i2string(ast_node["id"].get<std::int16_t>());
}

bool solidity_convertert::get_var_decl_name(
  const nlohmann::json &decl,
  std::string &name,
  std::string &id)
{
  std::string cname;
  get_current_contract_name(decl, cname);

  if (decl["stateVariable"])
    get_state_var_decl_name(decl, cname, name, id);
  else
  {
    if (cname.empty() && decl["mutability"] == "constant")
      // global variable
      get_state_var_decl_name(decl, "", name, id);
    else
      get_local_var_decl_name(decl, cname, name, id);
  }

  return false;
}

// parse the non-state variable
void solidity_convertert::get_local_var_decl_name(
  const nlohmann::json &ast_node,
  const std::string &cname,
  std::string &name,
  std::string &id)
{
  assert(ast_node.contains("id"));
  assert(ast_node.contains("name"));

  name = ast_node["name"].get<std::string>();
  if (current_functionDecl && !cname.empty() && !current_functionName.empty())
  {
    // converting local variable inside a function
    // For non-state functions, we give it different id.
    // E.g. for local variable i in function nondet(), it's "sol:@C@Base@F@nondet@i#55".

    // As the local variable inside the function will not be inherited, we can use current_functionName
    id = "sol:@C@" + cname + "@F@" + current_functionName + "@" + name + "#" +
         i2string(ast_node["id"].get<std::int16_t>());
  }
  else if (ast_node.contains("scope"))
  {
    // This means we are handling a local variable which is not inside a function body.
    //! Assume it is a variable inside struct/error which can be declared outside the contract
    int scp = ast_node["scope"].get<int>();
    if (member_entity_scope.count(scp) == 0)
    {
      log_error("cannot find struct/error name");
      abort();
    }
    std::string struct_name = member_entity_scope.at(scp);
    if (cname.empty())
      id = "sol:@" + struct_name + "@" + name + "#" +
           i2string(ast_node["id"].get<std::int16_t>());
    else
      id = "sol:@C@" + cname + "@" + struct_name + "@" + name + "#" +
           i2string(ast_node["id"].get<std::int16_t>());
  }
  else
  {
    log_error("Unsupported local variable");
    abort();
  }
}

void solidity_convertert::get_error_definition_name(
  const nlohmann::json &ast_node,
  std::string &name,
  std::string &id)
{
  std::string cname;
  get_current_contract_name(ast_node, cname);
  const int id_num = ast_node["id"].get<int>();
  name = ast_node["name"].get<std::string>();
  if (cname.empty())
    id = "sol:@" + name + "#" + std::to_string(id_num);
  else
    // e.g. sol:@C@Base@F@error@1
    id = "sol:@C@" + cname + "@F@" + name + "#" + std::to_string(id_num);
}

void solidity_convertert::get_function_definition_name(
  const nlohmann::json &ast_node,
  std::string &name,
  std::string &id)
{
  // Follow the way in clang:
  //  - For function name, just use the ast_node["name"]
  // assume Solidity AST json object has "name" field, otherwise throws an exception in nlohmann::json
  std::string contract_name;
  get_current_contract_name(ast_node, contract_name);
  if (contract_name.empty())
  {
    name = ast_node["name"].get<std::string>();
    id = "sol:@F@" + name + "#" + i2string(ast_node["id"].get<std::int16_t>());
    return;
  }

  //! for event/... who have added an body node. It seems that a ["kind"] is automatically added.?
  if (
    ast_node.contains("kind") && !ast_node["kind"].is_null() &&
    ast_node["kind"].get<std::string>() == "constructor")
    // In solidity
    // - constructor does not have a name
    // - there can be only one constructor in each contract
    // we, however, mimic the C++ grammar to manually assign it with a name
    // whichi is identical to the contract name
    // we also allows multiple constructor where the added ctor has no  `id`
    name = contract_name;
  else
    name = ast_node["name"] == "" ? ast_node["kind"] : ast_node["name"];

  id = "sol:@C@" + contract_name + "@F@" + name + "#" +
       i2string(ast_node["id"].get<std::int16_t>());

  log_debug("solidity", "\t\t@@@ got function name {}", name);
}

unsigned int solidity_convertert::add_offset(
  const std::string &src,
  unsigned int start_position)
{
  // extract the length from "start:length:index"
  std::string offset = src.substr(1, src.find(":"));
  // already added 1 in start_position
  unsigned int end_position = start_position + std::stoul(offset);
  return end_position;
}

// get the constructor symbol id
// noted that the ctor might not have been parsed yet
bool solidity_convertert::get_ctor_call_id(
  const std::string &contract_name,
  std::string &ctor_id)
{
  // we first try to find the explicit constructor defined in the source file.
  ctor_id = get_explicit_ctor_call_id(contract_name);
  if (ctor_id.empty())
    // then we try to find the implicit constructor we manually added
    ctor_id = get_implict_ctor_call_id(contract_name);

  if (ctor_id.empty())
  {
    // this means the neither explicit nor implicit constructor is found
    return true;
  }
  return false;
}

// get the explicit constructor symbol id
// retrun empty string if no explicit ctor
std::string
solidity_convertert::get_explicit_ctor_call_id(const std::string &contract_name)
{
  // get the constructor
  const nlohmann::json &ctor_ref = find_constructor_ref(contract_name);
  if (!ctor_ref.empty())
  {
    int id = ctor_ref["id"].get<int>();
    return "sol:@C@" + contract_name + "@F@" + contract_name + "#" +
           std::to_string(id);
  }

  // not found
  return "";
}

// get the implicit constructor symbol id
std::string
solidity_convertert::get_implict_ctor_call_id(const std::string &contract_name)
{
  // for implicit ctor, the id is manually set as 0
  return "sol:@C@" + contract_name + "@F@" + contract_name + "#";
}

std::string
solidity_convertert::get_src_from_json(const nlohmann::json &ast_node)
{
  // some nodes may have "src" inside a member json object
  // we need to deal with them case by case based on the node type
  SolidityGrammar::ExpressionT type =
    SolidityGrammar::get_expression_t(ast_node);
  switch (type)
  {
  case SolidityGrammar::ExpressionT::ImplicitCastExprClass:
  {
    assert(ast_node.contains("subExpr"));
    assert(ast_node["subExpr"].contains("src"));
    return ast_node["subExpr"]["src"].get<std::string>();
  }
  case SolidityGrammar::ExpressionT::NullExpr:
  {
    // empty address
    return "-1:-1:-1";
  }
  default:
  {
    log_error("Unsupported node type when getting src from JSON");
    abort();
  }
  }
}

unsigned int solidity_convertert::get_line_number(
  const nlohmann::json &ast_node,
  bool final_position)
{
  // Solidity src means "start:length:index", where "start" represents the position of the first char byte of the identifier.
  std::string src = ast_node.contains("src")
                      ? ast_node["src"].get<std::string>()
                      : get_src_from_json(ast_node);

  std::string position = src.substr(0, src.find(":"));
  unsigned int byte_position = std::stoul(position) + 1;

  if (final_position)
    byte_position = add_offset(src, byte_position);

  // the line number can be calculated by counting the number of line breaks prior to the identifier.
  unsigned int loc = std::count(
                       contract_contents.begin(),
                       (contract_contents.begin() + byte_position),
                       '\n') +
                     1;
  return loc;
}

void solidity_convertert::get_location_from_node(
  const nlohmann::json &ast_node,
  locationt &location)
{
  location.set_line(get_line_number(ast_node));
  location.set_file(
    absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory

  // To annotate local declaration within a function
  if (current_functionDecl)
  {
    location.set_function(
      current_functionName); // set the function where this local variable belongs to
  }
}

void solidity_convertert::get_start_location_from_stmt(
  const nlohmann::json &ast_node,
  locationt &location)
{
  std::string function_name;

  if (current_functionDecl)
    function_name = current_functionName;

  // The src manager of Solidity AST JSON is too encryptic.
  // For the time being we are setting it to "1".
  location.set_line(get_line_number(ast_node));
  location.set_file(
    absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory

  if (!function_name.empty())
    location.set_function(function_name);
}

void solidity_convertert::get_final_location_from_stmt(
  const nlohmann::json &ast_node,
  locationt &location)
{
  std::string function_name;

  if (current_functionDecl)
    function_name = current_functionName;

  // The src manager of Solidity AST JSON is too encryptic.
  // For the time being we are setting it to "1".
  location.set_line(get_line_number(ast_node, true));
  location.set_file(
    absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory

  if (!function_name.empty())
    location.set_function(function_name);
}

std::string solidity_convertert::get_modulename_from_path(std::string path)
{
  std::string filename = get_filename_from_path(path);

  if (filename.find_last_of('.') != std::string::npos)
    return filename.substr(0, filename.find_last_of('.'));

  return filename;
}

std::string solidity_convertert::get_filename_from_path(std::string path)
{
  if (path.find_last_of('/') != std::string::npos)
    return path.substr(path.find_last_of('/') + 1);

  return path; // for _x, it just returns "overflow_2.c" because the test program is in the same dir as esbmc binary
}

// Find the last parent json node
// It will not reliably find the correct parent if the same target appears under multiple different parent nodes.
// To enusre correctness, the input is expected to contain key "id" and, if possible, "is_inherit"
const nlohmann::json &solidity_convertert::find_last_parent(
  const nlohmann::json &root,
  const nlohmann::json &target)
{
  using Frame = const nlohmann::json *; // Pointer to a node
  std::stack<Frame> stack;
  stack.push(&root);

  while (!stack.empty())
  {
    const nlohmann::json *node = stack.top();
    stack.pop();

    if (node->is_object())
    {
      for (auto it = node->begin(); it != node->end(); ++it)
      {
        const auto &value = it.value();
        if (value == target)
          return *node;

        if (value.is_structured())
          stack.push(&value);
      }
    }
    else if (node->is_array())
    {
      for (const auto &element : *node)
      {
        if (element == target)
          return *node;

        if (element.is_structured())
          stack.push(&element);
      }
    }
  }

  return empty_json;
}

// return the parent contract definition node
// return empty_json if the target_json is outside of any contract
// this function dose not rely on current_baseContractName
// so we assume that the target provided is no ambiguous
const nlohmann::json &solidity_convertert::find_parent_contract(
  const nlohmann::json &root,
  const nlohmann::json &target)
{
  using Frame = std::pair<const nlohmann::json *, const nlohmann::json *>;
  std::stack<Frame> stack;
  // Begin with the root, with no current contract context.
  stack.emplace(&root, nullptr);

  while (!stack.empty())
  {
    auto [node, current_contract] = stack.top();
    stack.pop();

    if (
      node->is_object() && node->contains("nodeType") &&
      (*node)["nodeType"] == "ContractDefinition")
    {
      current_contract = node;
    }

    // check if this node is the target.
    // We use pointer identity comparison to ensure an exact match.
    if (*node == target)
    {
      return current_contract ? *current_contract : empty_json;
    }

    // If the node is an object, iterate over its values.
    if (node->is_object())
    {
      // Use reverse order for DFS consistency.
      for (auto it = node->rbegin(); it != node->rend(); ++it)
      {
        const auto &value = it.value();
        if (value.is_structured())
          stack.emplace(&value, current_contract);
      }
    }
    // If the node is an array, do the same.
    else if (node->is_array())
    {
      for (auto it = node->rbegin(); it != node->rend(); ++it)
      {
        if (it->is_structured())
          stack.emplace(&(*it), current_contract);
      }
    }
  }

  // If the target was not found, return an empty JSON.
  return empty_json;
}

// Searches for the target node inside a contract body.
// Assumes that the caller is already in the base contract node.
const nlohmann::json &solidity_convertert::find_decl_ref_in_contract(
  const nlohmann::json &j,
  int ref_id)
{
  // Check if this node matches the ref_id.
  // Skip any nested contract definition (should not occur).
  if (!j.is_structured())
    return empty_json;

  using Frame = const nlohmann::json *;
  std::stack<Frame> stack;
  stack.push(&j);

  while (!stack.empty())
  {
    const nlohmann::json *node = stack.top();
    stack.pop();

    if (node->is_object())
    {
      if (node->contains("id") && (*node)["id"] == ref_id)
      {
        log_debug("solidity", "\tfound");
        return *node;
      }

      for (auto it = node->rbegin(); it != node->rend(); ++it)
      {
        const auto &value = it.value();
        if (value.is_structured())
          stack.push(&value);
      }
    }
    else if (node->is_array())
    {
      for (auto it = node->rbegin(); it != node->rend(); ++it)
      {
        if (it->is_structured())
          stack.push(&(*it));
      }
    }
  }

  return empty_json;
}

// Searches for the target node at the global level.
// When a ContractDefinition is encountered, only the base contract (by name)
// is considered and, if matched, its body is searched using find_decl_ref_in_contract.
const nlohmann::json &
solidity_convertert::find_decl_ref_global(const nlohmann::json &j, int ref_id)
{
  if (!j.is_structured())
    return empty_json;

  using Frame = const nlohmann::json *;
  std::stack<Frame> stack;
  stack.push(&j);

  while (!stack.empty())
  {
    const nlohmann::json *node = stack.top();
    stack.pop();

    if (node->is_object())
    {
      // Check if the current object is a ContractDefinition.
      if (
        node->contains("nodeType") &&
        (*node)["nodeType"] == "ContractDefinition")
      {
        // we will not merge the contract-definition
        // so we are safe to compare the id here
        if (node->contains("id") && (*node)["id"] == ref_id)
          return *node;

        if (
          node->contains("contractKind") &&
          (*node)["contractKind"] == "library")
        {
          const nlohmann::json &result =
            find_decl_ref_in_contract(*node, ref_id);
          if (!result.is_null() && !result.empty())
            return result;
        }

        // This is a contract definition; only process if it is the base contract.
        if (
          node->contains("name") && !current_baseContractName.empty() &&
          (*node)["name"] == current_baseContractName)
        {
          // Search recursively in the contract body.
          const nlohmann::json &result =
            find_decl_ref_in_contract(*node, ref_id);
          if (!result.is_null() && !result.empty())
            return result;
        }

        // For contract definitions that are not the base, do not search inside.
        continue;
      }
      else
      {
        // For non-contract nodes at the global scope,
        // we can safely check if this node itself matches.
        if (node->contains("id") && (*node)["id"] == ref_id)
          return *node;
      }

      // Recurse on all children of this global-level object.
      for (auto it = node->rbegin(); it != node->rend(); ++it)
      {
        const auto &value = it.value();
        if (value.is_structured())
          stack.push(&value);
      }
    }
    else if (node->is_array())
    {
      for (auto it = node->rbegin(); it != node->rend(); ++it)
      {
        if (it->is_structured())
          stack.push(&(*it));
      }
    }
  }

  return empty_json;
}

// find json reference via id only
const nlohmann::json &solidity_convertert::find_decl_ref_unique_id(
  const nlohmann::json &j,
  int ref_id)
{
  return find_decl_ref_in_contract(j, ref_id);
}

// find the first (and the only) node with matched ref_id under based contract
// the searching range includes: global-definition + based-contract-definition
// e.g.
// base_contractName = "Base"
// file:
//    struct Test{}  <-- not inside any contract. we will search this
//    contract Base {}  <-- match. we will search it
//    contract Dereive{} <-- not match, not searching it
// The reason we specify base_contractName is that, the `id` is not unqiue any more after we merging the inherited nodes.
// return empty_json if not found
const nlohmann::json &
solidity_convertert::find_decl_ref(const nlohmann::json &j, int ref_id)
{
  log_debug(
    "solidity",
    "\tcurrent base contract name {}, ref_id {}",
    current_baseContractName,
    std::to_string(ref_id));
  return find_decl_ref_global(j, ref_id);
}

// return construcor node based on the *contract* id
const nlohmann::json &solidity_convertert::find_constructor_ref(int contract_id)
{
  nlohmann::json &nodes = src_ast_json["nodes"];
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    if (
      (*itr)["id"].get<int>() == contract_id &&
      (*itr)["nodeType"] == "ContractDefinition")
    {
      nlohmann::json &ast_nodes = (*itr)["nodes"];
      for (nlohmann::json::iterator ittr = ast_nodes.begin();
           ittr != ast_nodes.end();
           ++ittr)
      {
        if ((*ittr)["kind"] == "constructor")
          return *ittr;
      }
    }
  }

  // implicit constructor call
  return empty_json;
}

const nlohmann::json &
solidity_convertert::find_constructor_ref(const std::string &contract_name)
{
  log_debug(
    "solidity", "\t@@@ finding reference of constructor {}", contract_name);
  nlohmann::json &nodes = src_ast_json["nodes"];
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    if (
      (*itr).contains("name") &&
      (*itr)["name"].get<std::string>() == contract_name &&
      (*itr)["nodeType"] == "ContractDefinition")
    {
      nlohmann::json &ast_nodes = (*itr)["nodes"];
      for (nlohmann::json::iterator ittr = ast_nodes.begin();
           ittr != ast_nodes.end();
           ++ittr)
      {
        if ((*ittr)["kind"] == "constructor")
          return *ittr;
      }
    }
  }

  log_debug("solidity", "\t@@@ Failed to find explicit constructor");
  // implicit constructor call
  return empty_json;
}

void solidity_convertert::get_default_symbol(
  symbolt &symbol,
  std::string module_name,
  typet type,
  std::string name,
  std::string id,
  locationt location)
{
  symbol.mode = mode;
  symbol.module = module_name;
  symbol.location = std::move(location);
  symbol.type = std::move(type);
  symbol.name = name;
  symbol.id = id;
}

symbolt *solidity_convertert::move_symbol_to_context(symbolt &symbol)
{
  return context.move_symbol_to_context(symbol);
}

void solidity_convertert::convert_expression_to_code(exprt &expr)
{
  if (expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}

bool solidity_convertert::check_intrinsic_function(
  const nlohmann::json &ast_node)
{
  // function to detect special intrinsic functions, e.g. ___ESBMC_assume
  return (
    ast_node.contains("name") && (ast_node["name"] == "__ESBMC_assume" ||
                                  ast_node["name"] == "__VERIFIER_assume" ||
                                  ast_node["name"] == "__ESBMC_assert" ||
                                  ast_node["name"] == "__VERIFIER_assert"));
}

nlohmann::json solidity_convertert::make_implicit_cast_expr(
  const nlohmann::json &sub_expr,
  std::string cast_type)
{
  log_debug("solidity", "\t@@@ make_implicit_cast_expr");
  // Since Solidity AST does not have type cast information about return values,
  // we need to manually make a JSON object and wrap the return expression in it.
  std::map<std::string, std::string> m = {
    {"nodeType", "ImplicitCastExprClass"},
    {"castType", cast_type},
    {"subExpr", {}}};
  nlohmann::json implicit_cast_expr = m;
  implicit_cast_expr["subExpr"] = sub_expr;

  return implicit_cast_expr;
}

nlohmann::json
solidity_convertert::make_pointee_type(const nlohmann::json &sub_expr)
{
  // Since Solidity function call node does not have enough information, we need to make a JSON object
  // manually create a JSON object to complete the conversions of function to pointer decay

  // make a mapping for JSON object creation latter
  // based on the usage of get_func_decl_ref_t() in get_func_decl_ref_type()
  nlohmann::json adjusted_expr;

  if (
    sub_expr["typeString"].get<std::string>().find("function") !=
    std::string::npos)
  {
    // Add more special functions here
    if (
      sub_expr["typeString"].get<std::string>().find("function ()") !=
        std::string::npos ||
      sub_expr["typeIdentifier"].get<std::string>().find(
        "t_function_assert_pure$") != std::string::npos ||
      sub_expr["typeIdentifier"].get<std::string>().find(
        "t_function_internal_pure$") != std::string::npos)
    {
      // e.g. FunctionNoProto: "typeString": "function () returns (uint8)" with () empty after keyword 'function'
      // "function ()" contains the function args in the parentheses.
      // make a type to behave like SolidityGrammar::FunctionDeclRefT::FunctionNoProto
      // Note that when calling "assert(.)", it's like "typeIdentifier": "t_function_assert_pure$......",
      //  it's also treated as "FunctionNoProto".
      auto j2 = R"(
            {
              "nodeType": "FunctionDefinition",
              "parameters":
                {
                  "parameters" : []
                }
            }
          )"_json;
      adjusted_expr = j2;

      if (
        sub_expr["typeString"].get<std::string>().find("returns") !=
        std::string::npos)
      {
        adjusted_expr = R"(
            {
              "nodeType": "FunctionDefinition",
              "parameters":
                {
                  "parameters" : []
                }
            }
          )"_json;
        // e.g. for typeString like:
        // "typeString": "function () returns (uint8)"
        // use regex to capture the type and convert it to shorter form.
        std::smatch matches;
        std::regex e("returns \\((\\w+)\\)");
        std::string typeString = sub_expr["typeString"].get<std::string>();
        if (std::regex_search(typeString, matches, e))
        {
          auto j2 = nlohmann::json::parse(
            R"({
                "typeIdentifier": "t_)" +
            matches[1].str() + R"(",
                "typeString": ")" +
            matches[1].str() + R"("
              })");
          adjusted_expr["returnParameters"]["parameters"][0]
                       ["typeDescriptions"] = j2;
        }
        else if (
          sub_expr["typeString"].get<std::string>().find("returns (contract") !=
          std::string::npos)
        {
          // TODO: Fix me
          auto j2 = R"(
              {
                "nodeType": "ParameterList",
                "parameters": []
              }
            )"_json;
          adjusted_expr["returnParameters"] = j2;
        }
        else
          assert(!"Unsupported return types in pointee");
      }
      else
      {
        // e.g. for typeString like:
        // "typeString": "function (bool) pure"
        auto j2 = R"(
              {
                "nodeType": "ParameterList",
                "parameters": []
              }
            )"_json;
        adjusted_expr["returnParameters"] = j2;
      }
    }
    else
      assert(!"Unsupported - detected function call with parameters");
  }
  else
    assert(!"Unsupported pointee - currently we only support the semantics of function to pointer decay");

  return adjusted_expr;
}

// Parse typet object into a typeDescriptions json
nlohmann::json solidity_convertert::make_return_type_from_typet(typet type)
{
  // Useful to get the width of a int literal type for return statement
  nlohmann::json adjusted_expr;
  if (type.is_signedbv() || type.is_unsignedbv())
  {
    std::string width = type.width().as_string();
    std::string type_name = (type.is_signedbv() ? "int" : "uint") + width;
    auto j2 = nlohmann::json::parse(
      R"({
              "typeIdentifier": "t_)" +
      type_name + R"(",
              "typeString": ")" +
      type_name + R"("
            })");
    adjusted_expr = j2;
  }
  return adjusted_expr;
}

nlohmann::json solidity_convertert::make_array_elementary_type(
  const nlohmann::json &type_descrpt)
{
  // Function used to extract the type of the array and its elements
  // In order to keep the consistency and maximum the reuse of get_type_description function,
  // we used ["typeDescriptions"] instead of ["typeName"], despite the fact that the latter contains more information.
  // Although ["typeDescriptions"] also contains all the information needed, we have to do some pre-processing

  // e.g.
  //   "typeDescriptions": {
  //     "typeIdentifier": "t_array$_t_uint256_$dyn_memory_ptr",
  //     "typeString": "uint256[] memory"
  //      }
  //
  // convert to
  //
  //   "typeDescriptions": {
  //     "typeIdentifier": "t_uint256",
  //     "typeString": "uint256"
  //     }

  //! current implement does not consider Multi-Dimensional Arrays

  // 1. declare an empty json node
  nlohmann::json elementary_type;
  const std::string typeIdentifier =
    type_descrpt["typeIdentifier"].get<std::string>();

  // 2. extract type info
  // e.g.
  //  bytes[] memory x => "t_array$_t_bytes_$dyn_storage_ptr"  => t_bytes
  //  [1,2,3]          => "t_array$_t_uint8_$3_memory_ptr      => t_uint8
  assert(typeIdentifier.substr(0, 8) == "t_array$");
  std::regex rgx("\\$_\\w*_\\$");

  std::smatch match;
  if (!std::regex_search(typeIdentifier, match, rgx))
    assert(!"Cannot find array element type in typeIdentifier");
  std::string sub_match = match[0];
  std::string t_type = sub_match.substr(2, sub_match.length() - 4);
  std::string type = t_type.substr(2);

  // 3. populate node
  elementary_type = {{"typeIdentifier", t_type}, {"typeString", type}};

  return elementary_type;
}

nlohmann::json solidity_convertert::make_array_to_pointer_type(
  const nlohmann::json &type_descrpt)
{
  // Function to replace the content of ["typeIdentifier"] with "ArrayToPtr"
  // All the information in ["typeIdentifier"] should also be available in ["typeString"]
  std::string type_identifier = "ArrayToPtr";
  std::string type_string = type_descrpt["typeString"].get<std::string>();

  std::map<std::string, std::string> m = {
    {"typeIdentifier", type_identifier}, {"typeString", type_string}};
  nlohmann::json adjusted_type = m;

  return adjusted_type;
}

std::string
solidity_convertert::get_array_size(const nlohmann::json &type_descrpt)
{
  const std::string s = type_descrpt["typeString"].get<std::string>();
  std::regex rgx(".*\\[([0-9]+)\\]");
  std::string the_size;

  std::smatch match;
  if (std::regex_search(s.begin(), s.end(), match, rgx))
  {
    std::ssub_match sub_match = match[1];
    the_size = sub_match.str();
  }
  else
    assert(!"Unsupported - Missing array size in type descriptor. Detected dynamic array?");

  return the_size;
}

bool solidity_convertert::is_dyn_array(const nlohmann::json &ast_node)
{
  if (
    ast_node.contains("typeDescriptions") &&
    SolidityGrammar::get_type_name_t(ast_node["typeDescriptions"]) ==
      SolidityGrammar::DynArrayTypeName)
    return true;
  return false;
}

void solidity_convertert::get_size_of_expr(const typet &t, exprt &size_of_expr)
{
  size_of_expr = exprt("sizeof", size_type());
  typet elem_type = t;
  if (elem_type.is_struct())
  {
    struct_union_typet st = to_struct_union_type(elem_type);
    elem_type = symbol_typet(prefix + st.tag().as_string());
  }
  size_of_expr.set("#c_sizeof_type", elem_type);
}

// check if the abi.encodedSignature is the same
// note that internal/private function do not have abi signature
bool solidity_convertert::is_func_sig_cover(
  const std::string &derived,
  const std::string &base)
{
  // function signature coverage‐check lambda: name + ordered argument types
  auto covers =
    [&](const std::string &derived, const std::string &base) -> bool {
    const auto &dSigs = funcSignatures.at(derived);
    const auto &bSigs = funcSignatures.at(base);

    // Every base sig must have a matching derived sig
    for (const auto &ds : dSigs)
    {
      if (ds.name == derived)
        // skip ctor
        continue;

      if (ds.visibility == "private" || ds.visibility == "internal")
        // cannot be called via abi
        continue;

      //TODO: skip interface, abstract contract

      bool foundMatch = false;

      for (const auto &bs : bSigs)
      {
        // 1) same name?
        if (ds.name != bs.name)
          continue;

        // 2) internal or private?
        if (bs.visibility == "private" || bs.visibility == "internal")
          continue;

        // 3) same number of params?
        const auto &dArgs = to_code_type(ds.type).arguments();
        const auto &bArgs = to_code_type(bs.type).arguments();
        if (dArgs.size() != bArgs.size())
          continue;
        if (
          to_code_type(ds.type).has_ellipsis() &&
          !to_code_type(bs.type).has_ellipsis())
          continue;
        if (
          to_code_type(bs.type).has_ellipsis() &&
          !to_code_type(ds.type).has_ellipsis())
          continue;

        // 4) each parameter's type must match, in order
        bool argsMatch = true;
        for (size_t idx = 0; idx < dArgs.size(); ++idx)
        {
          if (dArgs[idx].type() != bArgs[idx].type())
          {
            argsMatch = false;
            break;
          }
        }

        if (argsMatch)
        {
          log_debug("solidity", "function {} matched", ds.name);
          foundMatch = true;
          break;
        }
      }

      // if any base‐fn had no match, derived does NOT cover base
      if (!foundMatch)
        return false;
    }

    return true;
  };

  return covers(derived, base);
}

// check if the target contract contains any public var with matched name and type, which can be accessed via abi
bool solidity_convertert::is_var_getter_matched(
  const std::string &cname,
  const std::string &tname,
  const typet &ttype)
{
  log_debug(
    "solidity",
    "heck if the target contract {} contains any public var {} with matched "
    "name and type",
    cname,
    tname);
  // 1) get contract body
  nlohmann::json contract_ref;
  for (auto &nodes : src_ast_json["nodes"])
  {
    if (
      nodes.contains("nodeType") && nodes["nodeType"] == "ContractDefinition" &&
      nodes["name"] == cname)
      contract_ref = nodes;
  }
  if (contract_ref.empty())
  {
    log_error("cannot find contract definition ref");
    abort();
  }

  const nlohmann::json body = contract_ref["nodes"];
  for (const auto &node : body)
  {
    if (
      SolidityGrammar::get_contract_body_element_t(node) ==
      SolidityGrammar::VarDecl)
    {
      assert(node.contains("visibility"));
      std::string access = node["visibility"].get<std::string>();
      if (access != "public")
        continue;

      exprt comp;
      if (get_var_decl_ref(node, false, comp))
      {
        log_error("failed to get variable reference");
        abort();
      }

      if (comp.name().as_string() == tname && comp.type() == ttype)
        return true;
    }
  }

  return false;
}

/**
 * @param decl_ref: the declaration of the ctor. Can be empty for implicit ctor.
 * @caller: the caller node that might contain the arguments
*/
bool solidity_convertert::get_ctor_call(
  const nlohmann::json &decl_ref,
  const nlohmann::json &caller,
  side_effect_expr_function_callt &call)
{
  /*
  we need to convert
    call(1)
  to
      call(&Base, 1)
  */
  log_debug("solidity", "\t\t@@@ get_ctor_call");
  locationt l;
  get_location_from_node(caller, l);

  if (!decl_ref.empty())
  {
    if (get_non_library_function_call(decl_ref, caller, call))
      return true;

    // reset the type. due to the empty returnParameters, the type of the call
    // is wrongly set as void.
    const auto &_contract =
      find_parent_contract(src_ast_json["nodes"], decl_ref);
    std::string c_name = _contract["name"].get<std::string>();
    call.type() = symbol_typet(prefix + c_name);

    // reset the this object
    exprt this_object;
    get_new_object(symbol_typet(prefix + c_name), this_object);
    call.arguments().at(0) = this_object;

    // set constructor
    call.set("constructor", 1);
  }
  else
  {
    log_error("unexpected implicit ctor");
    return true;
  }

  return false;
}

// wrapper
bool solidity_convertert::get_library_function_call(
  const nlohmann::json &decl_ref,
  const nlohmann::json &caller,
  side_effect_expr_function_callt &call)
{
  assert(!decl_ref.empty());
  assert(decl_ref.contains("returnParameters"));

  exprt func;
  if (get_func_decl_ref(decl_ref, func))
    return true;

  code_typet t;
  if (get_type_description(decl_ref["returnParameters"], t.return_type()))
    return true;

  return get_library_function_call(func, t, decl_ref, caller, call);
}

// library/error/event functions have no definition node
// the key difference comparing to the `get_non_library_function_call` is that we do not need a this-object as the first argument for the function call
// the key difference is that we do not add this pointer.
bool solidity_convertert::get_library_function_call(
  const exprt &func,
  const typet &t,
  const nlohmann::json &decl_ref,
  const nlohmann::json &caller,
  side_effect_expr_function_callt &call)
{
  call.function() = func;
  if (t.is_code())
  {
    call.type() = to_code_type(t).return_type();
    assert(!call.type().is_code());
  }
  else
    call.type() = t;
  locationt l;
  get_location_from_node(caller, l);
  call.location() = l;

  if (caller.contains("arguments"))
  {
    nlohmann::json param = nullptr;
    const nlohmann::json empty_array = nlohmann::json::array();
    auto itr = empty_array.end();
    auto itr_end = empty_array.end();
    if (!decl_ref.empty() && decl_ref.contains("parameters"))
    {
      assert(decl_ref["parameters"].contains("parameters"));
      const nlohmann::json &param_nodes = decl_ref["parameters"]["parameters"];
      itr = param_nodes.begin();
      itr_end = param_nodes.end();
    }

    //  builtin functions do not need the this object as the first arguments
    for (const auto &arg : caller["arguments"].items())
    {
      exprt single_arg;
      if (itr != itr_end && (*itr).contains("typeDescriptions"))
      {
        param = (*itr)["typeDescriptions"];
        ++itr;
      }
      else if (arg.value().contains("commonType"))
        param = arg.value()["commonType"];
      else if (arg.value().contains("typeDescriptions"))
        param = arg.value()["typeDescriptions"];

      if (get_expr(arg.value(), param, single_arg))
        return true;

      call.arguments().push_back(single_arg);
      param = nullptr;
    }
  }

  return false;
}

/** 
    * call to a non-library function:
    *   this.func(); // func(&this)
    * @param decl_ref: the function declaration node
    * @param caller: the function caller node which contains the arguments
    TODO: if the paramenter is a 'memory' type, we need to create
    a copy. E.g. string memory x => char *x => char * x_cpy
    this could be done by memcpy. However, for dyn_array, we do not have 
    the size info. Thus in the future we need to convert the dyn array to
    a struct which record both array and size. This will also help us to support
    array.length, .push and .pop 
**/
bool solidity_convertert::get_non_library_function_call(
  const nlohmann::json &decl_ref,
  const nlohmann::json &caller,
  side_effect_expr_function_callt &call)
{
  if (decl_ref.empty() || decl_ref.is_null())
  {
    log_error("unexpect empty or null declaration json");
    return true;
  }

  log_debug(
    "solidity",
    "\tget_non_library_function_call {}",
    decl_ref["name"] != "" ? decl_ref["name"].get<std::string>()
                           : decl_ref["kind"].get<std::string>());

  locationt loc = locationt();
  if (!caller.empty())
    get_location_from_node(caller, loc);
  else if (current_functionDecl)
    get_location_from_node(*current_functionDecl, loc);

  exprt func;
  if (get_func_decl_ref(decl_ref, func))
    return true;

  assert(decl_ref.contains("returnParameters"));
  code_typet t;
  if (get_type_description(decl_ref["returnParameters"], t.return_type()))
    return true;

  call.location() = loc;
  call.function() = func;
  call.function().location() = loc;
  call.type() = t.return_type();

  // Populating arguments

  // this object
  exprt this_object = nil_exprt();
  if (current_functionDecl)
  {
    if (get_func_decl_this_ref(*current_functionDecl, this_object))
      return true;
  }
  else if (!caller.empty())
  {
    if (get_ctor_decl_this_ref(caller, this_object))
      return true;
  }
  // otherwise, it's the auxiliary function we defined //e.g. call, delegatecall...

  call.arguments().push_back(this_object);

  if (decl_ref.contains("parameters") && caller.contains("arguments"))
  {
    // * Assume it is a normal funciton call, including ctor call with params
    // set caller object as the first argument

    nlohmann::json param_nodes = decl_ref["parameters"]["parameters"];
    nlohmann::json param = nullptr;
    nlohmann::json::iterator itr = param_nodes.begin();

    for (const auto &arg : caller["arguments"].items())
    {
      if (itr != param_nodes.end())
      {
        if ((*itr).contains("typeDescriptions"))
        {
          param = (*itr)["typeDescriptions"];
        }
        ++itr;
      }

      exprt single_arg;
      if (get_expr(arg.value(), param, single_arg))
        return true;

      call.arguments().push_back(single_arg);
      param = nullptr;
    }
  }
  else
  {
    // we know we are calling a function within the source code
    // however, the definition json or the calling argument json is not provided
    // it could be the function call in the multi-transaction-verification
    // populate nil arguements
    if (assign_param_nondet(decl_ref, call))
    {
      log_error("Failed to populate nil parameters");
      return true;
    }
  }

  return false;
}

// extract new contract instance expression
// we insert that contract name into the newContractSet if there is a new expresssion related to this contract
// e.g. Base x = new Base(); then we insert "Base" into newContractSet
// the idea is that if the contract is not used in 'new', then we can simply create a
// global static infinity array to play as a mapping structure
void solidity_convertert::extract_new_contracts()
{
  if (!src_ast_json.contains("nodes"))
    return;

  std::function<void(const nlohmann::json &)> process_node;
  process_node = [&](const nlohmann::json &node) {
    if (node.is_object())
    {
      if (node.contains("nodeType") && node["nodeType"] == "NewExpression")
      {
        if (node.contains("typeName"))
        {
          typet new_type;
          if (get_type_description(
                node["typeName"]["typeDescriptions"], new_type))
          {
            log_error("failed to obtain typeDescriptions");
            abort();
          }
          if (new_type.get("#sol_type") == "CONTRACT")
          {
            std::string contract_name = new_type.get("#sol_contract").c_str();
            newContractSet.insert(contract_name);
          }
        }
      }

      for (const auto &item : node.items())
      {
        process_node(item.value());
      }
    }
    else if (node.is_array())
    {
      for (const auto &child : node)
      {
        process_node(child);
      }
    }
  };

  for (const auto &top_level_node : src_ast_json["nodes"])
  {
    if (
      top_level_node.contains("nodeType") &&
      top_level_node["nodeType"] == "ContractDefinition")
    {
      // for get_type_descriptions
      std::string old = current_baseContractName;
      current_baseContractName = top_level_node["name"].get<std::string>();
      process_node(top_level_node);
      current_baseContractName = old;
    }
  }
}

/** 
 return the new-object expression
 basically we need to
 - get the ctor call expr
 - construct a "temporary_object" and set the ctor call as the operands
 @ast_node: caller node whose nodeType is NewExpression
*/
bool solidity_convertert::get_new_object_ctor_call(
  const nlohmann::json &caller,
  exprt &new_expr)
{
  log_debug("solidity", "generating new contract object");
  // 1. get the ctor call expr
  nlohmann::json callee_expr_json = caller["expression"];
  int ref_decl_id = callee_expr_json["typeName"]["referencedDeclaration"];

  // get contract name
  const std::string contract_name = contractNamesMap[ref_decl_id];
  if (contract_name.empty())
  {
    log_error("cannot find the contract name");
    abort();
  }
  if (get_new_object_ctor_call(contract_name, caller, new_expr))
    return true;

  return false;
}

// return a new expression: new Base(2);
bool solidity_convertert::get_new_object_ctor_call(
  const std::string &contract_name,
  const nlohmann::json caller,
  exprt &new_expr)
{
  log_debug("solidity", "get_new_object_ctor_call");
  assert(linearizedBaseList.count(contract_name) && !contract_name.empty());

  // setup initializer, i.e. call the constructor
  side_effect_expr_function_callt call;
  const nlohmann::json &constructor_ref = find_constructor_ref(contract_name);
  if (constructor_ref.empty())
    return get_implicit_ctor_ref(contract_name, new_expr);

  if (get_ctor_call(constructor_ref, caller, call))
    return true;

  // construct temporary object
  get_temporary_object(call, new_expr);
  return false;
}

bool solidity_convertert::get_implicit_ctor_ref(
  const std::string &contract_name,
  exprt &new_expr)
{
  log_debug("solidity", "\t\tget_implicit_ctor_ref");

  // to obtain the type info
  std::string name, id;
  name = contract_name;
  id = get_implict_ctor_call_id(contract_name);
  if (context.find_symbol(id) == nullptr)
  {
    if (add_implicit_constructor(contract_name))
      return true;
  }

  exprt ctor = symbol_expr(*context.find_symbol(id));
  code_typet type;
  type.return_type() = symbol_typet(prefix + contract_name);

  side_effect_expr_function_callt call;
  call.function() = ctor;
  call.set("constructor", 1);
  call.type() = type.return_type();
  call.location().file(absolute_path);

  exprt this_object;
  get_new_object(symbol_typet(prefix + contract_name), this_object);
  call.arguments().push_back(this_object);

  get_temporary_object(call, new_expr);
  return false;
}

/*
  e.g. x = func();
  ==>  x = nondet();
       unbound();
*/
bool solidity_convertert::get_unbound_funccall(
  const std::string contractName,
  code_function_callt &call)
{
  log_debug("solidity", "\tget_unbound_funccall");
  symbolt f_sym;
  if (get_unbound_function(contractName, f_sym))
    return true;

  code_function_callt func_call;
  func_call.location() = f_sym.location;
  func_call.function() = symbol_expr(f_sym);

  call = func_call;
  return false;
}

void solidity_convertert::get_static_contract_instance_name(
  const std::string c_name,
  std::string &name,
  std::string &id)
{
  name = "_ESBMC_Object_" + c_name;
  id = "sol:@" + name + "#";
}

void solidity_convertert::get_static_contract_instance_ref(
  const std::string &c_name,
  exprt &new_expr)
{
  std::string name, id;
  get_static_contract_instance_name(c_name, name, id);
  typet t = symbol_typet(prefix + c_name);
  new_expr = exprt("symbol", t);
  new_expr.identifier(id);
  new_expr.cmt_lvalue(true);
  new_expr.name(name);
  new_expr.pretty_name(name);
}

void solidity_convertert::add_static_contract_instance(const std::string c_name)
{
  log_debug("solidity", "\tAdd static instance of contract {}", c_name);

  std::string ctor_ins_name, ctor_ins_id;
  get_static_contract_instance_name(c_name, ctor_ins_name, ctor_ins_id);

  // inheritance instance: make a copy of the static instance
  // this is to make sure we insert it before the _ESBMC_Object_cname
  locationt ctor_ins_loc;
  ctor_ins_loc.file(absolute_path);
  ctor_ins_loc.line(1);
  std::string ctor_ins_debug_modulename =
    get_modulename_from_path(absolute_path);
  typet ctor_ins_typet = symbol_typet(prefix + c_name);

  symbolt ctor_ins_symbol;
  get_default_symbol(
    ctor_ins_symbol,
    ctor_ins_debug_modulename,
    ctor_ins_typet,
    ctor_ins_name,
    ctor_ins_id,
    ctor_ins_loc);
  ctor_ins_symbol.lvalue = true;
  ctor_ins_symbol.is_extern = false;
  // the instance should be set as static
  ctor_ins_symbol.static_lifetime = true;
  ctor_ins_symbol.file_local = true;

  auto &added_sym = *move_symbol_to_context(ctor_ins_symbol);

  // get value
  std::string ctor_id;
  // we do not check the return value as we might have not parsed the symbol yet
  if (get_ctor_call_id(c_name, ctor_id))
  {
    // probably the contract is not parsed yet
    log_error("cannot find ctor id for contract {}", c_name);
    abort();
  }
  if (context.find_symbol(ctor_id) == nullptr)
  {
    // this means that contract, including ctor, has not been parse yet
    // this will lead to issue in the following process, particuarly this_pointer
    const auto &json = find_constructor_ref(c_name);
    if (json.empty())
    {
      if (add_implicit_constructor(c_name))
      {
        log_error("Failed to add the implicit constructor");
        abort();
      }
    }
    else
    {
      // parsing the constructor in another contract
      std::string old = current_baseContractName;
      current_baseContractName = c_name;
      if (get_function_definition(json))
      {
        auto name = json["name"].get<std::string>();
        if (name.empty())
          name = json["kind"].get<std::string>();
        log_error(
          "Failed to parse the function {}", json["name"].get<std::string>());
        abort();
      }
      current_baseContractName = old;
    }
  }

  exprt ctor;
  if (get_new_object_ctor_call(c_name, empty_json, ctor))
  {
    log_error("failed to construct a temporary object");
    abort();
  }

  added_sym.value = ctor;
}

void solidity_convertert::get_inherit_static_contract_instance_name(
  const std::string bs_c_name,
  const std::string c_name,
  std::string &name,
  std::string &id)
{
  name = "_ESBMC_aux_" + c_name;
  id = "sol:@C@" + bs_c_name + "@" + name + "#";
}

void solidity_convertert::get_inherit_ctor_definition_name(
  const std::string c_name,
  std::string &name,
  std::string &id)
{
  name = c_name;
  id = "sol:@C@" + c_name + "@F@" + name + "#0";
}

void solidity_convertert::get_inherit_ctor_definition(
  const std::string c_name,
  exprt &new_expr)
{
  std::string fname, fid;
  get_inherit_ctor_definition_name(c_name, fname, fid);

  if (context.find_symbol(fid) != nullptr)
  {
    new_expr = symbol_expr(*context.find_symbol(fid));
    return;
  }
  code_typet ft;
  typet tmp_rtn_type("constructor");
  ft.return_type() = tmp_rtn_type;
  ft.set("#member_name", prefix + c_name);
  ft.set("#inlined", true);
  symbolt fs;
  locationt l;
  l.file(absolute_path);
  l.line(1);
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(fs, debug_modulename, ft, fname, fid, l);
  auto &add_sym = *move_symbol_to_context(fs);

  get_function_this_pointer_param(c_name, fid, debug_modulename, l, ft);

  // copy the param list
  // e.g. the original ctor: Base(int x)
  //      the inherit ctor:  Base(int x, bool aux)
  auto ctor_json = find_constructor_ref(c_name);
  bool is_implicit_ctor = ctor_json.empty();

  auto old_current_functionName = current_functionName;
  auto old_current_functionDecl = current_functionDecl;
  ctor_json["id"] = 0;
  ctor_json["is_inherited"] = true;
  current_functionDecl = &ctor_json;
  current_functionName = c_name;

  if (!is_implicit_ctor)
  {
    SolidityGrammar::ParameterListT params =
      SolidityGrammar::get_parameter_list_t(ctor_json["parameters"]);
    if (params != SolidityGrammar::ParameterListT::EMPTY)
    {
      for (const auto &decl : ctor_json["parameters"]["parameters"].items())
      {
        const nlohmann::json &func_param_decl = decl.value();

        code_typet::argumentt param;
        if (get_function_params(func_param_decl, c_name, param))
        {
          log_error("internal error in parsing params");
          abort();
        }
        ft.arguments().push_back(param);
      }
    }
  }

  // param: bool
  std::string aname = "aux";
  std::string aid = "sol:@C@" + c_name + "@F@" + c_name + "@" + aname + "#";
  typet addr_t = bool_type();
  addr_t.cmt_constant(true);
  symbolt addr_s;
  get_default_symbol(addr_s, debug_modulename, addr_t, aname, aid, l);
  move_symbol_to_context(addr_s);

  auto param = code_typet::argumentt();
  param.type() = addr_t;
  param.cmt_base_name(aname);
  param.cmt_identifier(aid);
  param.location() = l;
  ft.arguments().push_back(param);
  add_sym.type = ft;

  // body
  exprt func_body = code_blockt();
  if (!is_implicit_ctor)
  {
    // insert
    for (auto &node : src_ast_json["nodes"])
    {
      if (node["nodeType"] == "ContractDefinition" && node["name"] == c_name)
        node["nodes"].push_back(ctor_json);
    }

    if (
      (*current_functionDecl).contains("body") ||
      ((*current_functionDecl).contains("implemented") &&
       (*current_functionDecl)["implemented"] == true))
    {
      log_debug(
        "solidity", "\t parsing function {}'s body", current_functionName);
      if (get_block((*current_functionDecl)["body"], func_body))
      {
        log_error("internal error in constructing inherit_ctor body");
        abort();
      }
    }

    // remove insertion
    for (nlohmann::json &node : src_ast_json["nodes"])
    {
      if (node["nodeType"] == "ContractDefinition" && node["name"] == c_name)
      {
        // Reverse iterate to find the last inserted object with id == 0
        for (auto it = node.rbegin(); it != node.rend(); ++it)
        {
          if (
            it.value().is_object() && it.value().contains("id") &&
            it.value()["id"].get<int>() == 0)
          {
            // Erase requires a forward iterator, so convert it
            node.erase(std::next(it).base());
            break; // Only one needs to be removed
          }
        }
      }
    }
  }

  // reset
  current_functionDecl = old_current_functionDecl;
  current_functionName = old_current_functionName;

  add_sym.value = func_body;

  new_expr = symbol_expr(add_sym);
  move_builtin_to_contract(c_name, new_expr, "public", true);
}

// create an const instance only used for inheritance assignment
// e.g. this->x = _ESBMC_ctor_A_tmp.x;
// @bs_c_name: caller's contract name
// @c_name: callee ctor name
void solidity_convertert::get_inherit_static_contract_instance(
  const std::string bs_c_name,
  const std::string c_name,
  const nlohmann::json &args_list,
  symbolt &sym)
{
  log_debug("solidity", "get inherit_static_contract_instance");
  std::string ctor_ins_name, ctor_ins_id;
  get_inherit_static_contract_instance_name(
    bs_c_name, c_name, ctor_ins_name, ctor_ins_id);

  if (context.find_symbol(ctor_ins_id) != nullptr)
  {
    sym = *context.find_symbol(ctor_ins_id);
    return;
  }

  std::string ctor_ins_debug_modulename =
    get_modulename_from_path(absolute_path);
  typet ctor_Ins_typet = symbol_typet(prefix + c_name);

  symbolt ctor_ins_symbol;
  get_default_symbol(
    ctor_ins_symbol,
    ctor_ins_debug_modulename,
    ctor_Ins_typet,
    ctor_ins_name,
    ctor_ins_id,
    locationt());
  ctor_ins_symbol.lvalue = true;
  ctor_ins_symbol.file_local = true;

  symbolt &added_ctor_symbol = *move_symbol_to_context(ctor_ins_symbol);

  // create aux constructor
  // Base(&this, float y) // since solidity has no floating point
  side_effect_expr_function_callt call;
  exprt new_expr;
  get_inherit_ctor_definition(c_name, new_expr);

  exprt this_object;
  get_new_object(symbol_typet(prefix + c_name), this_object);
  call.arguments().push_back(this_object);

  if (args_list.contains("arguments"))
  {
    // get param type
    auto &decl_ref = find_constructor_ref(c_name);
    if (decl_ref.empty() || decl_ref.is_null())
    {
      log_error("cannot find ctor ref");
      abort();
    }

    assert(decl_ref.contains("parameters"));
    nlohmann::json param_nodes = decl_ref["parameters"]["parameters"];
    nlohmann::json param = nullptr;
    nlohmann::json::iterator itr = param_nodes.begin();

    for (const auto &arg : args_list["arguments"].items())
    {
      if (itr != param_nodes.end())
      {
        if ((*itr).contains("typeDescriptions"))
        {
          param = (*itr)["typeDescriptions"];
        }
        ++itr;
      }

      exprt single_arg;
      if (get_expr(arg.value(), param, single_arg))
      {
        log_error("parsing arguments error");
        abort();
      }

      call.arguments().push_back(single_arg);
      param = nullptr;
    }
  }

  call.arguments().push_back(true_exprt());

  call.function() = new_expr;
  call.set("constructor", 1);
  call.type() = symbol_typet(prefix + c_name);
  call.location().file(absolute_path);
  call.location().line(1);
  exprt val;
  get_temporary_object(call, val);
  added_ctor_symbol.value = val;
  sym = added_ctor_symbol;
}

void solidity_convertert::get_contract_mutex_name(
  const std::string c_name,
  std::string &name,
  std::string &id)
{
  name = "$mutex_" + c_name;
  id = "sol:@C@" + c_name + "@" + name + "#";
}

// for reentry check
void solidity_convertert::get_contract_mutex_expr(
  const std::string c_name,
  const exprt &this_expr,
  exprt &expr)
{
  std::string name, id;
  exprt _mutex;
  get_contract_mutex_name(c_name, name, id);
  if (context.find_symbol(id) == nullptr)
  {
    log_error("cannot find auxiliary var {}", id);
    abort();
  }
  _mutex = symbol_expr(*context.find_symbol(id));

  expr = member_exprt(this_expr, _mutex.name(), _mutex.type());
}

bool solidity_convertert::is_sol_builin_symbol(
  const std::string &cname,
  const std::string &name)
{
  std::string tx_name, tx_id;
  get_contract_mutex_name(cname, tx_name, tx_id);
  std::set<std::string> list = {
    "$address", "$codehash", "$balance", "$code", tx_name, "_ESBMC_bind_cname"};
  if (list.count(name) != 0)
    return true;

  return false;
}

/*
  old_sender = msg.sender
  msg.sender = instance.$address
  (_ESBMC_mutext_Base = true)
  (call to payable func)
  (_ESBMC_mutext_Base = false)
  msg.sender = old_sender;
  */
bool solidity_convertert::get_high_level_call_wrapper(
  const std::string cname,
  const exprt &this_expr,
  exprt &front_block,
  exprt &back_block)
{
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  exprt msg_sender = symbol_expr(*context.find_symbol("c:@msg_sender"));

  typet addr_t = unsignedbv_typet(160);
  addr_t.set("#sol_type", "ADDRESS");
  symbolt old_sender;
  get_default_symbol(
    old_sender,
    debug_modulename,
    addr_t,
    "old_sender",
    "sol:@C@" + cname + "@F@old_sender#" + std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_sender = *move_symbol_to_context(old_sender);
  code_declt old_sender_decl(symbol_expr(added_old_sender));
  added_old_sender.value = msg_sender;
  old_sender_decl.operands().push_back(msg_sender);
  front_block.move_to_operands(old_sender_decl);

  // msg_sender = this.address;
  exprt this_address = member_exprt(this_expr, "$address", addr_t);
  exprt assign_sender = side_effect_exprt("assign", addr_t);
  assign_sender.copy_to_operands(msg_sender, this_address);
  convert_expression_to_code(assign_sender);
  front_block.move_to_operands(assign_sender);

  if (is_reentry_check)
  {
    exprt _mutex;
    get_contract_mutex_expr(cname, this_expr, _mutex);

    // _ESBMC_mutex = true;
    typet _t = bool_type();
    _t.set("#sol_type", "BOOL");
    _t.set("#cpp_type", "bool");
    exprt _true = true_exprt();
    exprt _false = false_exprt();
    _true.location() = this_expr.location();
    _false.location() = this_expr.location();
    exprt assign_lock = side_effect_exprt("assign", _t);
    assign_lock.copy_to_operands(_mutex, _true);
    assign_lock.location() = this_expr.location();
    convert_expression_to_code(assign_lock);
    front_block.move_to_operands(assign_lock);

    // _ESBMC_mutex = false;
    exprt assign_unlock = side_effect_exprt("assign", _t);
    assign_unlock.copy_to_operands(_mutex, _false);
    assign_unlock.location() = this_expr.location();
    convert_expression_to_code(assign_unlock);
    back_block.move_to_operands(assign_unlock);
  }

  // msg_sender = old_sender;
  exprt assign_sender_restore = side_effect_exprt("assign", addr_t);
  assign_sender_restore.copy_to_operands(
    msg_sender, symbol_expr(added_old_sender));
  convert_expression_to_code(assign_sender_restore);
  back_block.move_to_operands(assign_sender_restore);
  return false;
}

bool solidity_convertert::is_bytes_type(const typet &t)
{
  if (t.get("#sol_type").as_string().find("BYTES") != std::string::npos)
    return true;
  return false;
}

void solidity_convertert::convert_type_expr(
  const namespacet &ns,
  exprt &src_expr,
  const typet &dest_type)
{
  log_debug("solidity", "\t@@@ Performing type conversion");

  typet src_type = src_expr.type();
  // only do conversion when the src.type != dest.type
  if (src_type != dest_type)
  {
    std::string src_sol_type = src_type.get("#sol_type").as_string();
    std::string dest_sol_type = dest_type.get("#sol_type").as_string();

    log_debug("solidity", "\t\tGot src_sol_type = {}", src_sol_type);
    log_debug("solidity", "\t\tGot dest_sol_type = {}", dest_sol_type);

    if (
      (dest_sol_type == "ADDRESS" || dest_sol_type == "ADDRESS_PAYABLE") &&
      (src_sol_type == "CONTRACT" || src_sol_type.empty()))
    {
      // CONTRACT: address(instance) ==> instance.address
      // EMPTY: address(this) ==> this.address
      std::string comp_name = "$address";
      typet t = unsignedbv_typet(160);
      t.set("#sol_type", "ADDRESS");

      src_expr = member_exprt(src_expr, comp_name, t);
    }
    else if (
      (src_sol_type == "ADDRESS" || src_sol_type == "ADDRESS_PAYABLE") &&
      dest_sol_type == "CONTRACT")
    {
      // E.g. for `Derive x = Derive(_addr)`:
      // => Derive* x = &_ESBMC_Obeject_Derive;
      // because in trusted mode, the address has been limited to the set of _ESBMC_Object

      exprt c_ins;
      std::string _cname = dest_type.get("#sol_contract").as_string();
      get_static_contract_instance_ref(_cname, c_ins);

      // type conversion
      src_expr = c_ins;
    }
    else if (is_bytes_type(src_type) && is_bytes_type(dest_type))
    {
      // 1. Fixed-size Bytes Converted to Smaller Types
      //    bytes2 a = 0x4326;
      //    bytes1 b = bytes1(a); // b will be 0x43
      // 2. Fixed-size Bytes Converted to Larger Types
      //    bytes2 a = 0x4235;
      //    bytes4 b = bytes4(a); // b will be 0x42350000
      // which equals to:
      //    new_type b = bswap(new_type)(bswap(x)))

      exprt bswap_expr, sub_bswap_expr;

      // 1. bswap
      sub_bswap_expr = exprt("bswap", src_type);
      sub_bswap_expr.operands().push_back(src_expr);

      // 2. typecast
      solidity_gen_typecast(ns, sub_bswap_expr, dest_type);

      // 3. bswap back
      bswap_expr = exprt("bswap", sub_bswap_expr.type());
      bswap_expr.operands().push_back(sub_bswap_expr);

      src_expr = bswap_expr;
    }
    else if (
      (src_sol_type == "ARRAY_LITERAL") && src_type.id() == typet::id_array)
    {
      // this means we are handling a src constant array
      // which should be assigned to an array pointer
      // e.g. data1 = [int8(6), 7, -8, 9, 10, -12, 12];

      log_debug("solidity", "\t@@@ Converting array literal to symbol");

      if (dest_type.id() != typet::id_pointer)
      {
        log_error(
          "Expecting dest_type to be pointer type, got = {}",
          dest_type.id().as_string());
        abort();
      }

      // dynamic: uint x[] = [1,2]
      // fixed:   uint x[3] = [1,2], whose rhs array is incomplete and need to add zero element
      // the goal is to convert the rhs constant array to a static global var

      // get rhs constant array size
      const std::string src_size = src_type.get("#sol_array_size").as_string();
      if (src_size.empty())
      {
        // e.g. a = new uint[](len);
        // we have already populate the auxiliary state var so
        // skip the rest of the process
        // ? solidity_gen_typecast(ns, src_expr, dest_type);
        return;
      }
      unsigned z_src_size = std::stoul(src_size, nullptr);

      // get lhs array size
      std::string dest_size = dest_type.get("#sol_array_size").as_string();
      if (dest_size.empty())
      {
        if (dest_sol_type == "ARRAY")
        {
          log_error("Unexpected empty-length fixed array");
          abort();
        }
        // the dynamic array does not have a fixed length
        // therefore set it as the rhs length
        dest_size = src_size;
      }
      unsigned z_dest_size = std::stoul(dest_size, nullptr);
      constant_exprt dest_array_size = constant_exprt(
        integer2binary(z_dest_size, bv_width(int_type())),
        integer2string(z_dest_size),
        int_type());

      if (src_expr.id() == irept::id_member)
      {
        // e.g. uint[3] x;  (x, y) = ([1,z], ...)
        // where [1,2] ==> uint8[] ==> tuple_instance.mem0
        // ==>
        //  x  = [（uint256)tuple_instance.mem0[0], （uint256)tuple_instance.mem0[1], 0]
        // - src_expr: [1, z]
        // - dest_type: uint*
        array_typet arr_t = array_typet(dest_type.subtype(), dest_array_size);
        exprt new_arr = exprt(irept::id_array, arr_t);

        exprt arr_comp;
        for (unsigned i = 0; i < z_src_size; i++)
        {
          // do array index
          exprt idx = constant_exprt(
            integer2binary(i, bv_width(size_type())),
            integer2string(i),
            size_type());
          exprt op = index_exprt(src_expr, idx, src_type.subtype());

          arr_comp = typecast_exprt(op, dest_type.subtype());
          new_arr.operands().push_back(arr_comp);
        }

        src_expr = new_arr;
        src_type = new_arr.type();
      }

      // allow fall-through
      if (src_expr.id() == irept::id_array)
      {
        log_debug("solidity", "\t@@@ Populating zero elements to array");

        // e.g. uint[3] x = [1] ==> uint[3] x == [1,0,0]
        unsigned s_size = src_expr.operands().size();
        if (s_size != z_src_size)
        {
          log_error(
            "Expecting equivalent array size, got {} and {}",
            std::to_string(s_size),
            std::to_string(z_src_size));
          abort();
        }
        if (z_dest_size > s_size)
        {
          for (unsigned i = 0; i < s_size; i++)
          {
            exprt &op = src_expr.operands().at(i);
            solidity_gen_typecast(ns, op, dest_type.subtype());
          }
          exprt _zero =
            gen_zero(get_complete_type(dest_type.subtype(), ns), true);
          _zero.location() = src_expr.location();
          _zero.set("#cformat", 0);
          // push zero
          for (unsigned i = s_size; i < z_dest_size; i++)
            src_expr.operands().push_back(_zero);

          // reset size
          assert(src_expr.type().is_array());
          to_array_type(src_expr.type()).size() = dest_array_size;

          // update "#sol_array_size"
          src_expr.type().set("#sol_array_size", dest_size);
        }
      }

      // since it's a array-constant/string-constant, we could safely make it to a local var
      // this local var will not be referred again so the name could be random.
      // e.g.
      // int[3] p = [1,2];
      // => int *p = [1,2,3];
      // => static int[3] tmp1 = [1,2,3];
      // return: src_expr = symbol_expr(tmp1)
      exprt new_expr;
      get_aux_array(src_expr, new_expr);
      src_expr = new_expr;
    }
    else
      solidity_gen_typecast(ns, src_expr, dest_type);
  }
}

static inline void static_lifetime_init(const contextt &context, codet &dest)
{
  dest = code_blockt();

  // call designated "initialization" functions
  context.foreach_operand_in_order([&dest](const symbolt &s) {
    if (s.type.initialization() && s.type.is_code())
    {
      code_function_callt function_call;
      function_call.function() = symbol_expr(s);
      dest.move_to_operands(function_call);
    }
  });
}

void solidity_convertert::get_aux_var(
  std::string &aux_name,
  std::string &aux_id)
{
  do
  {
    aux_name = "_ESBMC_aux" + std::to_string(aux_counter);
    aux_id = "sol:@" + aux_name;
    ++aux_counter;
  } while (context.find_symbol(aux_id) != nullptr);
}

void solidity_convertert::get_aux_function(
  std::string &aux_name,
  std::string &aux_id)
{
  do
  {
    aux_name = "_ESBMC_aux" + std::to_string(aux_counter);
    aux_id = "sol:@F@" + aux_name;
    ++aux_counter;
  } while (context.find_symbol(aux_id) != nullptr);
}

void solidity_convertert::get_aux_array_name(
  std::string &aux_name,
  std::string &aux_id)
{
  do
  {
    aux_name = "aux_array" + std::to_string(aux_counter);
    aux_id = "sol:@" + aux_name;
    ++aux_counter;
  } while (context.find_symbol(aux_id) != nullptr);
}

void solidity_convertert::get_aux_array(const exprt &src_expr, exprt &new_expr)
{
  if (src_expr.name().as_string().find("aux_array") != std::string::npos)
  {
    // skip if it's already a aux array
    new_expr = src_expr;
    return;
  }
  std::string aux_name;
  std::string aux_id;
  get_aux_array_name(aux_name, aux_id);

  locationt loc = src_expr.location();
  std::string debug_modulename =
    get_modulename_from_path(loc.file().as_string());

  typet t = src_expr.type();
  t.set("#sol_type", "ARRAY");

  symbolt sym;
  get_default_symbol(sym, debug_modulename, t, aux_name, aux_id, loc);
  sym.static_lifetime = true;
  sym.is_extern = false;
  sym.lvalue = true;
  sym.file_local = true;

  symbolt &added_symbol = *move_symbol_to_context(sym);

  added_symbol.value = src_expr;
  new_expr = symbol_expr(added_symbol);
}

void solidity_convertert::get_size_expr(const exprt &rhs, exprt &size_expr)
{
  typet rt = rhs.type();

  unsigned int arr_size = 0;
  if (!rt.get("#sol_array_size").empty())
    arr_size = std::stoi(rt.get("#sol_array_size").as_string());
  else if (rt.has_subtype() && !rt.subtype().get("#sol_array_size").empty())
    arr_size = std::stoi(rt.subtype().get("#sol_array_size").as_string());
  else
  {
    // arr_size = _ESBMC_array_length(rhs);
    side_effect_expr_function_callt length_expr;
    get_library_function_call_no_args(
      "_ESBMC_array_length",
      "c:@F@_ESBMC_array_length",
      uint_type(),
      rhs.location(),
      length_expr);
    length_expr.arguments().push_back(rhs);
    size_expr = length_expr;

    // not fall through
    return;
  }

  size_expr = constant_exprt(
    integer2binary(arr_size, bv_width(uint_type())),
    integer2string(arr_size),
    uint_type());
}

void solidity_convertert::store_update_dyn_array(
  const exprt &dyn_arr,
  const exprt &size_expr,
  exprt &store_call)
{
  // void _ESBMC_store_array(void *array, size_t length)
  side_effect_expr_function_callt length_expr;
  get_library_function_call_no_args(
    "_ESBMC_store_array",
    "c:@F@_ESBMC_store_array",
    empty_typet(),
    dyn_arr.location(),
    length_expr);
  length_expr.arguments().push_back(dyn_arr);
  length_expr.arguments().push_back(size_expr);
  store_call = length_expr;
}

// convert new array rhs
// e.g. uint* x = calloc();
bool solidity_convertert::get_empty_array_ref(
  const nlohmann::json &expr,
  exprt &new_expr)
{
  // Get Name
  nlohmann::json callee_expr_json = expr["expression"];
  nlohmann::json callee_arg_json = expr["arguments"][0];

  // Get name, id;
  std::string name, id;
  get_aux_array_name(name, id);

  // Get Location
  locationt location_begin;
  get_location_from_node(callee_expr_json, location_begin);

  // Get Debug Module Name
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  // Get Type
  // 1. get elem type
  typet elem_type;
  const nlohmann::json elem_node =
    callee_expr_json["typeName"]["baseType"]["typeDescriptions"];
  if (get_type_description(elem_node, elem_type))
    return true;

  // 2. get array size
  exprt size;
  const nlohmann::json literal_type = callee_arg_json["typeDescriptions"];
  if (get_expr(callee_arg_json, literal_type, size))
    return true;

  // 3. do calloc
  side_effect_expr_function_callt calc_call;
  get_calloc_function_call(location_begin, calc_call);

  exprt size_of_expr;
  get_size_of_expr(elem_type, size_of_expr);

  calc_call.arguments().push_back(size);
  calc_call.arguments().push_back(size_of_expr);
  new_expr = calc_call;

  new_expr.type().set("#sol_type", "NEW_ARRAY");

  return false;
}

/*
  perform multi-transaction verification
  the idea is to verify the assertions that must be held 
  in any function calling order.
  convert the verifying contract to a "sol_main" function, e.g.

  Contract Base             
  {
      constrcutor(){}
      function A(){}
      function B(){}
  }

  will be converted to

  void sol_main()
  {
    Base()  // constructor_call
    while(nondet_bool)
    {
      if(nondet_bool) A();
      if(nondet_bool) B();
    }
  }

  Additionally, we need to handle the inheritance. Theoretically, we need to merge (i.e. create a copy) the public and internal state variables and functions inside Base contracts into the Derive contract. However, in practice we do not need to do so. Instead, we 
    - call the constructors based on the linearizedBaseList 
    - add the inherited public function call to the if-body 

*/
bool solidity_convertert::multi_transaction_verification(
  const std::string &c_name)
{
  log_debug(
    "Solidity", "@@@ performs transaction verification on contract {}", c_name);

  // 0. initialize "sol_main" body and while-loop body
  codet func_body, while_body;
  static_lifetime_init(context, func_body);
  func_body.make_block();
  static_lifetime_init(context, while_body);
  while_body.make_block();

  // add __ESBMC_HIDE
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  // initialize

  // 1. get constructor call
  if (contractNamesList.count(c_name) == 0)
  {
    log_error("Cannot find the contract name {}", c_name);
    return true;
  }

  // get sol harness function and move into the while body
  code_function_callt func_call;
  if (get_unbound_funccall(c_name, func_call))
    return true;
  while_body.move_to_operands(func_call);

  // while-cond:
  side_effect_expr_function_callt cond_expr = nondet_bool_expr;

  // while-loop statement:
  code_whilet code_while;
  code_while.cond() = cond_expr;
  code_while.body() = while_body;
  func_body.move_to_operands(code_while);

  // construct _ESBMC_Main_Base
  symbolt new_symbol;
  code_typet main_type;
  const std::string main_name = "_ESBMC_Main_" + c_name;
  const std::string main_id = "sol:@C@" + c_name + "@F@" + main_name + "#";
  typet e_type = empty_typet();
  e_type.set("cpp_type", "void");
  main_type.return_type() = e_type;
  const symbolt &_contract = *context.find_symbol(prefix + c_name);
  new_symbol.location = _contract.location;
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(
    new_symbol,
    debug_modulename,
    main_type,
    main_name,
    main_id,
    _contract.location);

  new_symbol.lvalue = true;
  new_symbol.is_extern = false;
  new_symbol.file_local = false;

  symbolt &main_sym = *context.move_symbol_to_context(new_symbol);

  // no params
  main_type.make_ellipsis();

  main_sym.type = main_type;
  main_sym.value = func_body;

  // set "_ESBMC_Main_X" as the main function
  // this will be overwrite in multi-contract mode.
  config.main = main_name;

  return false;
}

/*
  This function perform multi-transaction verification on each contract in isolation.
  To do so, we construct nondetered switch_case;
*/
bool solidity_convertert::multi_contract_verification_bound(
  std::set<std::string> &tgt_set)
{
  log_debug("solidity", "multi_contract_verification_bound");
  // 0. initialize "sol_main" body and switch body
  codet func_body, switch_body;
  static_lifetime_init(context, switch_body);
  static_lifetime_init(context, func_body);

  switch_body.make_block();
  func_body.make_block();

  // add __ESBMC_HIDE
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  // initialize

  // 1. construct switch-case
  int cnt = 0;
  std::set<std::string> cname_set;
  if (!tgt_set.empty())
    cname_set = tgt_set;
  else
    cname_set = contractNamesList;

  for (const auto &c_name : cname_set)
  {
    // 1.1 construct multi-transaction verification entry function
    // function "_ESBMC_Main_contractname" will be created and inserted to the symbol table.
    if (multi_transaction_verification(c_name))
    {
      log_error(
        "Failed to construct multi-transaction verification for contract {}",
        c_name);
      return true;
    }

    // 1.2 construct a "case n"
    exprt case_cond = constant_exprt(
      integer2binary(cnt, bv_width(int_type())),
      integer2string(cnt),
      int_type());

    // 1.3 construct case body: entry function + break
    codet case_body;
    static_lifetime_init(context, case_body);
    case_body.make_block();

    // func_call: _ESBMC_Main_contractname
    const std::string sub_sol_id =
      "sol:@C@" + c_name + "@F@_ESBMC_Main_" + c_name + "#";
    if (context.find_symbol(sub_sol_id) == nullptr)
    {
      log_error("Cannot find the {}'s main function {}", c_name, sub_sol_id);
      return true;
    }

    const symbolt &func = *context.find_symbol(sub_sol_id);
    code_function_callt func_expr;
    func_expr.location() = func.location;
    func_expr.function() = symbol_expr(func);
    case_body.move_to_operands(func_expr);

    // break statement
    exprt break_expr = code_breakt();
    case_body.move_to_operands(break_expr);

    // 1.4 construct case statement
    code_switch_caset switch_case;
    switch_case.case_op() = case_cond;
    convert_expression_to_code(case_body);
    switch_case.code() = to_code(case_body);

    // 1.5 move to switch body
    switch_body.move_to_operands(switch_case);

    // update case number counter
    ++cnt;
  }

  // 2. move switch to func_body
  // 2.1 construct nondet_uint jump condition
  side_effect_expr_function_callt cond_expr = nondet_uint_expr;

  // 2.2 construct switch statement
  code_switcht code_switch;
  code_switch.value() = cond_expr;
  code_switch.body() = switch_body;
  func_body.move_to_operands(code_switch);

  // 3. add "sol_main" to symbol table
  symbolt new_symbol;
  code_typet main_type;
  typet e_type = empty_typet();
  e_type.set("cpp_type", "void");
  main_type.return_type() = e_type;
  std::string sol_id;
  if (!tgt_set.empty())
    sol_id = "sol:@C@" + *tgt_set.begin() + "@F@_ESBMC_Main#";
  else
    sol_id = "sol:@F@_ESBMC_Main#";
  const std::string sol_name = "_ESBMC_Main";

  if (context.find_symbol(prefix + *contractNamesList.begin()) == nullptr)
  {
    log_error("Cannot find the main function");
    return true;
  }
  // use first contract's location
  const symbolt &contract =
    *context.find_symbol(prefix + *contractNamesList.begin());
  new_symbol.location = contract.location;
  std::string debug_modulename =
    get_modulename_from_path(contract.location.file().as_string());
  get_default_symbol(
    new_symbol,
    debug_modulename,
    main_type,
    sol_name,
    sol_id,
    new_symbol.location);

  new_symbol.lvalue = true;
  new_symbol.is_extern = false;
  new_symbol.file_local = false;

  symbolt &added_symbol = *context.move_symbol_to_context(new_symbol);

  // no params
  main_type.make_ellipsis();

  added_symbol.type = main_type;
  added_symbol.value = func_body;
  config.main = sol_name;
  return false;
}

// for unbound, we verify each contract individually
bool solidity_convertert::multi_contract_verification_unbound(
  std::set<std::string> &tgt_set)
{
  log_debug("solidity", "multi_contract_verification_unbound");
  codet func_body;
  static_lifetime_init(context, func_body);
  func_body.make_block();

  // add __ESBMC_HIDE
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  // initialize

  std::set<std::string> cname_set;
  if (!tgt_set.empty())
    cname_set = tgt_set;
  else
    cname_set = contractNamesList;

  for (const auto &c_name : cname_set)
  {
    // construct multi-transaction verification entry function
    // function "_ESBMC_Main_contractname" will be created and inserted to the symbol table.
    if (multi_transaction_verification(c_name))
    {
      log_error(
        "Failed to construct multi-transaction verification for contract {}",
        c_name);
      return true;
    }

    // func_call: _ESBMC_Main_contractname
    const std::string sub_sol_id =
      "sol:@C@" + c_name + "@F@_ESBMC_Main_" + c_name + "#";
    if (context.find_symbol(sub_sol_id) == nullptr)
    {
      log_error("Cannot find the {}'s main function {}", c_name, sub_sol_id);
      return true;
    }

    const symbolt &func = *context.find_symbol(sub_sol_id);
    code_function_callt func_expr;
    func_expr.location() = func.location;
    func_expr.function() = symbol_expr(func);
    func_body.move_to_operands(func_expr);
  }

  // add "sol_main" to symbol table
  symbolt new_symbol;
  code_typet main_type;
  typet e_type = empty_typet();
  e_type.set("cpp_type", "void");
  main_type.return_type() = e_type;
  const std::string sol_id = "sol:@F@_ESBMC_Main#";
  const std::string sol_name = "_ESBMC_Main";

  if (context.find_symbol(prefix + *contractNamesList.begin()) == nullptr)
  {
    log_error("Cannot find the main function");
    return true;
  }
  // use first contract's location
  const symbolt &contract =
    *context.find_symbol(prefix + *contractNamesList.begin());
  new_symbol.location = contract.location;
  std::string debug_modulename =
    get_modulename_from_path(contract.location.file().as_string());
  get_default_symbol(
    new_symbol,
    debug_modulename,
    main_type,
    sol_name,
    sol_id,
    new_symbol.location);

  new_symbol.lvalue = true;
  new_symbol.is_extern = false;
  new_symbol.file_local = false;

  symbolt &added_symbol = *context.move_symbol_to_context(new_symbol);

  // no params
  main_type.make_ellipsis();

  added_symbol.type = main_type;
  added_symbol.value = func_body;
  config.main = sol_name;

  return false;
}

bool solidity_convertert::is_low_level_call(const std::string &name)
{
  std::set<std::string> llc_set = {
    "call", "delegatecall", "staticcall", "callcode", "transfer", "send"};
  if (llc_set.count(name) != 0)
    return true;

  return false;
}

bool solidity_convertert::is_low_level_property(const std::string &name)
{
  std::set<std::string> llc_set = {"code", "codehash", "balance"};
  if (llc_set.count(name) != 0)
    return true;

  return false;
}

// e.g. address(x).balance => x->balance
// address(x).transfer() => x->transfer();

// everytime we call a ctor, we will assign an unique random address
// constructor(address _addr)
// {
//    A x = A(_addr);
// }
// =>
//  A tmp = new A();
//  if(_ESBMC_get_addr_array_idx(_addr) == -1)
//     tmp = &(struct A*)get_address_object_ptr(_addr);
// A& x = tmp;

bool solidity_convertert::add_auxiliary_members(const std::string contract_name)
{
  log_debug("solidity", "@@@ adding esbmc auxiliary members");

  // name prefix:
  std::string sol_prefix = "sol:@C@" + contract_name + "@";

  // value
  side_effect_expr_function_callt _ndt_uint = nondet_uint_expr;

  // _ESBMC_get_unique_address(this, cname)
  side_effect_expr_function_callt _addr;
  locationt l;
  l.function(contract_name);

  typet t;
  t = unsignedbv_typet(160);
  t.set("#sol_type", "ADDRESS");

  get_library_function_call_no_args(
    "_ESBMC_get_unique_address", "c:@F@_ESBMC_get_unique_address", t, l, _addr);

  exprt this_ptr;
  std::string ctor_id;
  get_ctor_call_id(contract_name, ctor_id);

  if (get_func_decl_this_ref(contract_name, ctor_id, this_ptr))
    return true;
  _addr.arguments().push_back(this_ptr);
  string_constantt cname_str(contract_name);
  _addr.arguments().push_back(cname_str);

  // address
  get_builtin_symbol(
    "$address", sol_prefix + "$address", t, l, _addr, contract_name);

  // codehash
  get_builtin_symbol(
    "$codehash",
    sol_prefix + "$codehash",
    unsignedbv_typet(256),
    l,
    _ndt_uint,
    contract_name);
  // balance
  get_builtin_symbol(
    "$balance",
    sol_prefix + "$balance",
    unsignedbv_typet(256),
    l,
    _ndt_uint,
    contract_name);
  // code
  get_builtin_symbol(
    "$code",
    sol_prefix + "$code",
    unsignedbv_typet(256),
    l,
    _ndt_uint,
    contract_name);

  if (is_reentry_check)
  {
    // populate reentry mutex flag
    std::string tx_name, tx_id;
    get_contract_mutex_name(contract_name, tx_name, tx_id);
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    typet _t = bool_type();
    _t.set("#sol_type", "BOOL");
    _t.set("#cpp_type", "bool");

    get_builtin_symbol(tx_name, tx_id, _t, l, gen_zero(_t), contract_name);
  }

  // binding
  exprt bind_expr;
  if (!is_bound)
  {
    string_constantt string(contract_name);
    bind_expr = string;
  }
  else
  {
    exprt call;
    if (assign_nondet_contract_name(contract_name, call))
      return true;
    bind_expr = call;
  }

  t = pointer_typet(signed_char_type());
  //t.set("#sol_type", "STRING");
  get_builtin_symbol(
    "_ESBMC_bind_cname",
    sol_prefix + "_ESBMC_bind_cname",
    t,
    l,
    bind_expr,
    contract_name);

  get_builtin_symbol(
    "_ESBMC_bind_cname",
    sol_prefix + "_ESBMC_bind_cname",
    t,
    l,
    bind_expr,
    contract_name);

  if (populate_low_level_functions(contract_name))
    return true;

  return false;
}

void solidity_convertert::move_builtin_to_contract(
  const std::string cname,
  const exprt &sym,
  bool is_method)
{
  move_builtin_to_contract(cname, sym, "private", is_method);
}

void solidity_convertert::move_builtin_to_contract(
  const std::string cname,
  const exprt &sym,
  const std::string &access,
  bool is_method)
{
  std::string c_id = prefix + cname;
  if (context.find_symbol(c_id) == nullptr)
  {
    log_error("parsing order error for struct {}", c_id);
    abort();
  }
  symbolt &c_sym = *context.find_symbol(c_id);
  assert(c_sym.type.is_struct());

  if (!is_method)
  {
    // check if it's already inserted
    for (auto i : to_struct_type(c_sym.type).components())
    {
      if (i.identifier() == sym.identifier())
        return;
    }

    struct_typet::componentt comp(sym.name(), sym.name(), sym.type());
    comp.set_access(access);
    comp.set("#lvalue", 1);
    comp.type().set("#member_name", c_sym.type.tag());
    to_struct_type(c_sym.type).components().push_back(comp);
  }
  else
  {
    // check if it's already inserted
    for (auto i : to_struct_type(c_sym.type).methods())
    {
      if (i.identifier() == sym.identifier())
        return;
    }

    struct_typet::componentt comp;
    // construct comp
    comp.type() = sym.type();
    comp.identifier(sym.identifier());
    comp.name(sym.name());
    comp.pretty_name(sym.name());
    comp.set_access(access);
    comp.id("symbol");
    to_struct_type(c_sym.type).methods().push_back(comp);
  }
}

// this funciton:
// - move the created auxiliary variables to the constructor
// - append the symbol as the component to the struct class
void solidity_convertert::get_builtin_symbol(
  const std::string name,
  const std::string id,
  const typet t,
  const locationt &l,
  const exprt &val,
  const std::string c_name)
{
  // skip if it's already in the symbol table
  if (context.find_symbol(id) != nullptr)
    return;

  symbolt sym;
  get_default_symbol(sym, "C++", t, name, id, l);
  sym.type.set("#sol_state_var", "1");
  sym.file_local = true;
  sym.lvalue = true;
  auto &added_sym = *move_symbol_to_context(sym);
  code_declt decl(symbol_expr(added_sym));
  added_sym.value = val;
  decl.operands().push_back(val);
  move_to_initializer(decl);

  if (!c_name.empty())
    // we need to update the fields of the contract struct symbol
    move_builtin_to_contract(c_name, symbol_expr(added_sym), false);
}

bool solidity_convertert::get_new_temporary_obj(
  const std::string &c_name,
  const std::string &ctor_ins_name,
  const std::string &ctor_ins_id,
  const locationt &ctor_ins_loc,
  symbolt &added,
  codet &decl)
{
  log_debug("solidity", "get new temporary object for contract {}", c_name);
  std::string ctor_ins_debug_modulename =
    get_modulename_from_path(absolute_path);
  typet ctor_Ins_typet = symbol_typet(prefix + c_name);

  symbolt ctor_ins_symbol;
  get_default_symbol(
    ctor_ins_symbol,
    ctor_ins_debug_modulename,
    ctor_Ins_typet,
    ctor_ins_name,
    ctor_ins_id,
    ctor_ins_loc);

  // since we might re-entry the sol_main_i, we
  // should set it as static global var
  ctor_ins_symbol.static_lifetime = true;
  ctor_ins_symbol.lvalue = true;
  ctor_ins_symbol.file_local = true;

  symbolt &added_ctor_symbol = *move_symbol_to_context(ctor_ins_symbol);

  exprt ctor;
  if (get_new_object_ctor_call(c_name, empty_json, ctor))
  {
    log_error("Failed in get_new_object_ctor_call");
    return true;
  }

  code_declt _decl(symbol_expr(added_ctor_symbol));
  added_ctor_symbol.value = ctor;
  _decl.operands().push_back(ctor);

  added = added_ctor_symbol;
  decl = _decl;
  return false;
}

// create a function: get_{property_name}(addr)
// this function is universal for every contract
void solidity_convertert::get_aux_property_function(
  const std::string &cname,
  const exprt &base,
  const typet &return_t,
  const locationt &loc,
  const std::string &property_name,
  exprt &new_expr)
{
  std::string fname = "get_" + property_name;
  std::string fid = "sol:@C@" + cname + "@F@" + fname + "#";

  exprt cur_this_expr;
  if (current_functionDecl)
  {
    if (get_func_decl_this_ref(*current_functionDecl, cur_this_expr))
      abort();
  }
  else
  {
    const auto &ctor_json = find_constructor_ref(cname);
    if (get_func_decl_this_ref(ctor_json, cur_this_expr))
      abort();
  }

  assert(base.is_constant() || base.is_member() || base.is_symbol());

  if (context.find_symbol(fid) != nullptr)
  {
    side_effect_expr_function_callt _call;
    get_library_function_call_no_args(fname, fid, return_t, loc, _call);
    _call.arguments().push_back(cur_this_expr);
    _call.arguments().push_back(base);
    new_expr = _call;
    return;
  }

  // poplate function definition
  // e.g. get_balance(this, this->addr);
  symbolt sym;
  code_typet type;
  type.return_type() = return_t;
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(sym, debug_modulename, type, fname, fid, loc);
  auto &added_symbol = *move_symbol_to_context(sym);

  get_function_this_pointer_param(cname, fid, debug_modulename, loc, type);

  // param: arg
  std::string aname = "_addr";
  std::string aid = "sol:@F@" + fname + "@" + aname + "#";
  typet addr_t = unsignedbv_typet(160);
  addr_t.set("#sol_type", "ADDRESS");
  addr_t.cmt_constant(true);
  symbolt addr_s;
  get_default_symbol(addr_s, debug_modulename, addr_t, aname, aid, loc);
  move_symbol_to_context(addr_s);

  auto param = code_typet::argumentt();
  param.type() = addr_t;
  param.cmt_base_name(aname);
  param.cmt_identifier(aid);
  param.location() = loc;
  type.arguments().push_back(param);

  // populate param
  added_symbol.type = type;
  // move to struct symbol
  move_builtin_to_contract(cname, symbol_expr(added_symbol), true);

  exprt addr_expr = symbol_expr(*context.find_symbol(aid));
  /* body:
      address(_addr).code
    =>
      if(get_object(_addr, "A") != NULL)
        return  (A *)get_object(_addr, "A")->code;
      if(get_object(_addr, "B") != NULL)
        return  (B *)get_object(_addr, "B")->code;
      return nondet_uint();
  */

  code_blockt _block;

  // hide it
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  _block.move_to_operands(label);

  for (auto cname : contractNamesList)
  {
    if (context.find_symbol("c:@F@_ESBMC_get_obj") == nullptr)
    {
      log_error("cannot find builtin library");
      abort();
    }

    // param
    string_constantt _cname(cname);

    // get_object(_addr, "A")
    side_effect_expr_function_callt get_obj;
    get_library_function_call_no_args(
      "_ESBMC_get_obj",
      "c:@F@_ESBMC_get_obj",
      pointer_typet(empty_typet()),
      loc,
      get_obj);

    get_obj.arguments().push_back(addr_expr);
    get_obj.arguments().push_back(_cname);

    // typecast
    typet _struct = symbol_typet(prefix + cname);
    exprt tc = typecast_exprt(get_obj, pointer_typet(_struct));

    // member access
    std::string comp_name = "$" + property_name;
    exprt mem = member_exprt(tc, comp_name, return_t);

    // return
    code_returnt ret_call;
    ret_call.return_value() = mem;

    // if(get_object(_addr, "A") != NULL)
    exprt _null = gen_zero(pointer_typet(empty_typet()));
    exprt _equal = exprt("notequal", bool_type());
    _equal.operands().push_back(get_obj);
    _equal.operands().push_back(_null);
    _equal.location() = loc;

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, ret_call);
    if_expr.location() = loc;
    _block.move_to_operands(if_expr);
  }

  // return nondet_uint
  code_returnt ret_uint;
  ret_uint.return_value() = nondet_uint_expr;
  _block.move_to_operands(ret_uint);

  // populate body
  added_symbol.value = _block;

  // do function call
  side_effect_expr_function_callt _call;
  _call.function() = symbol_expr(added_symbol);
  _call.location() = loc;
  _call.arguments().push_back(cur_this_expr);
  _call.arguments().push_back(base);
  new_expr = _call;
}

// get member access of built-in property.
// e.g. x.$balance, x.$code ...
void solidity_convertert::get_builtin_property_expr(
  const std::string &cname,
  const std::string &name,
  const exprt &base,
  const locationt &loc,
  exprt &new_expr)
{
  log_debug("solidity", "Getting built-in property");

  typet t;
  std::string comp_name = "$" + name;

  if (name == "address")
  {
    t = unsignedbv_typet(160);
    t.set("#sol_type", "ADDRESS");
  }
  else if (name == "code" || name == "codehash" || name == "balance")
  {
    t = unsignedbv_typet(256);
    //? set sol_type?
  }
  else
  {
    log_error("got unexpected builtin property {}", name);
    abort();
  }

  exprt mem;
  if (
    base.is_member() && (base.op0().name() == "this" ||
                         base.op0().type().get("#sol_type") == "CONTRACT"))
    // e.g. address(_ins_).balance => _ins_.balance
    //      address(this) => this->address
    //TODO: fixme! this pattern match is weak
    mem = member_exprt(base.op0(), comp_name, t);
  else
  {
    // e.g. address(msg.sender).balance
    // we do not know what instance is msg.sender pointed to, so over-approximate
    get_aux_property_function(cname, base, t, loc, name, mem);
  }

  mem.location() = loc;
  new_expr = mem;
}

bool solidity_convertert::get_base_contract_name(
  const exprt &base,
  std::string &cname)
{
  log_debug("solidity", "\t\t@@@ get_base_contract_name");

  if (base.type().get("#sol_contract").as_string().empty())
  {
    log_error("cannot find base contract name");
    return true;
  }

  cname = base.type().get("#sol_contract").as_string();
  return false;
}

void solidity_convertert::get_nondet_expr(const typet &t, exprt &new_expr)
{
  new_expr = exprt("sideeffect", t);
  new_expr.statement("nondet");
}

// x._ESBMC_bind_cname = _ESBMC_get_nondet_cname();
//                        ^^^^^^^^^^^^^^^^^^^^^^^^
// for high-level call, we bind the external calls with cname
// e.g.
// if(x.cname == Base)
//   _ESBMC_Object_Base.func()
// for low-level call, we bind the external calls with address
bool solidity_convertert::assign_nondet_contract_name(
  const std::string &_cname,
  exprt &new_expr)
{
  locationt l;
  l.function(_cname);

  side_effect_expr_function_callt _call;
  get_library_function_call_no_args(
    "_ESBMC_get_nondet_cont_name",
    "c:@F@_ESBMC_get_nondet_cont_name",
    pointer_typet(signed_char_type()),
    l,
    _call);

  std::unordered_set<std::string> cname_set;
  unsigned int length = 0;

  cname_set = structureTypingMap[_cname];
  length = cname_set.size();
  assert(!cname_set.empty());

  exprt size_expr;
  size_expr = constant_exprt(
    integer2binary(length, bv_width(uint_type())),
    integer2string(length),
    uint_type());

  // convert this string array (e.g. {"base", "derive"}) to a symbol
  std::string aux_name, aux_id;
  aux_name = "$" + _cname + "_bind_cname_list";
  aux_id = "sol:@C@" + _cname + "@" + aux_name;

  if (context.find_symbol(aux_id) == nullptr)
  {
    log_error("cannot find contract cname list");
    return true;
  }
  exprt sym = symbol_expr(*context.find_symbol(aux_id));

  // _ESBMC_get_nondet_cont_name
  _call.arguments().push_back(sym);
  _call.arguments().push_back(size_expr);

  new_expr = _call;
  return false;
}

// special handle for the contract type parameter.
// we need to bind them to the contract instance _ESBMC_Object
bool solidity_convertert::assign_param_nondet(
  const nlohmann::json &decl_ref,
  side_effect_expr_function_callt &call)
{
  log_debug("solidity", "\t\tpopulating nil arguments");
  assert(decl_ref.contains("parameters"));

  nlohmann::json param_nodes = decl_ref["parameters"]["parameters"];
  unsigned int cnt = 1;
  for (const auto &p_node : param_nodes)
  {
    if (p_node.contains("typeDescriptions"))
    {
      typet t;
      if (get_type_description(p_node["typeDescriptions"], t))
        return true;
      if (t.get("#sol_type") == "CONTRACT")
      {
        /*
            e.g. function run(Base x)
            ==>
            if(nondet_bool())
            {
              __ESBMC_Object_m.run(_ESBMC_Object_Base) 
              / / where its cname = ["Base", "Derive"]
            }
          */
        std::string base_cname = t.get("#sol_contract").as_string();
        assert(!base_cname.empty());
        exprt s;
        get_static_contract_instance_ref(base_cname, s);
        call.arguments().push_back(s);
      }
      else if (t.get("#sol_type") == "STRING" && is_pointer_check)
      {
        //! specific for string, we need to explicitly assign it as nondet_string()
        // otherwise we will get invalid_object
        side_effect_expr_function_callt nondet_str;
        get_library_function_call_no_args(
          "nondet_string",
          "c:@F@nondet_string",
          pointer_typet(signed_char_type()),
          locationt(),
          nondet_str);
        call.arguments().push_back(nondet_str);
      }
      else
        call.arguments().push_back(static_cast<const exprt &>(get_nil_irep()));
    }
    ++cnt;
  }

  return false;
}

// check if the target contract have at least one non-ctor external or public function
bool solidity_convertert::has_callable_func(const std::string &cname)
{
  return std::any_of(
    funcSignatures[cname].begin(),
    funcSignatures[cname].end(),
    [&cname](const solidity_convertert::func_sig &sig) {
      // must be public or external, even if the address is itself
      return sig.name != cname &&
             (sig.visibility == "public" || sig.visibility == "external");
    });
}

// check if there is a function with `func_name` in the contract `cname`
bool solidity_convertert::has_target_function(
  const std::string &cname,
  const std::string func_name)
{
  auto it = funcSignatures.find(cname);
  if (it == funcSignatures.end())
    return false;

  return std::any_of(
    it->second.begin(), it->second.end(), [&](const func_sig &sig) {
      return sig.name == func_name;
    });
}

solidity_convertert::func_sig solidity_convertert::get_target_function(
  const std::string &cname,
  const std::string &func_name)
{
  // Check if the contract exists in funcSignatures
  auto it = funcSignatures.find(cname);
  if (it == funcSignatures.end())
  {
    // If contract not found, return an empty func_sig
    return solidity_convertert::func_sig(
      "", "", "", code_typet(), false, false, false);
  }

  // Search for the function with the matching name
  auto &functions = it->second;
  auto func_it = std::find_if(
    functions.begin(),
    functions.end(),
    [&func_name](const solidity_convertert::func_sig &sig) {
      return sig.name == func_name;
    });

  // If function is found, return it; otherwise, return an empty func_sig
  if (func_it != functions.end())
  {
    return *func_it;
  }
  else
  {
    return solidity_convertert::func_sig(
      "",
      "",
      "",
      code_typet(),
      false,
      false,
      false); // Return an empty func_sig if not found
  }
}

bool solidity_convertert::get_high_level_member_access(
  const nlohmann::json &expr,
  const exprt &base,
  const exprt &member,
  const exprt &_mem_call,
  const bool is_func_call,
  exprt &new_expr)
{
  return get_high_level_member_access(
    expr, empty_json, base, member, _mem_call, is_func_call, new_expr);
}

/** 
 * Conversion: 
  constructor()
  {
    this->_ESBMC_bind_cname = get_nondet_cname(); // unless we have a new Base(), then = Base;
  }

  function test1(Base x, address _addr) public
  {
      x = new Base();     // x._ESBMC_bind_cname = Base;
      x.test();           // if x._ESBMC_bind_cname == base
                          //   _ESBMC_Object_base.test();
      x = Base(_addr);    // x = ESBMC_Object_base
                          // if _addr == _ESBMC_Object_base.$address
                          //   x._ESBMC_bind_cname = base
                          // if _addr == _ESBMC_Object_y.$address
                          //   x._ESBMC_bind_cname = y;
  }	

  the auxilidary tmp var will not be created if the member_type is void
  @expr: the whole member access expression json
  @options: call with options
  @is_func_call: true if it's a function member access; false state variable access
  @_mem_call: function call statement, with arguments populated
  return true: we fail to generate the high_level_member_access bound harness
               however, this should not be treated as an erorr.
               E.g. x.access() where x is a state variable
*/
bool solidity_convertert::get_high_level_member_access(
  const nlohmann::json &expr,
  const nlohmann::json &options,
  const exprt &base,
  const exprt &member,
  const exprt &_mem_call,
  const bool is_func_call,
  exprt &new_expr)
{
  log_debug("solidity", "Getting high-level member access");
  new_expr = _mem_call;

  // get 'Base'
  std::string _cname;
  if (base.type().get("#sol_type") != "CONTRACT")
  {
    log_error("Expecting contract type");
    base.type().dump();
    return true;
  }
  if (get_base_contract_name(base, _cname))
    return true;

  // current contract name
  std::string cname;
  get_current_contract_name(expr, cname);

  // current this pointer reference
  exprt cur_this_expr;
  if (current_functionDecl)
  {
    if (get_func_decl_this_ref(*current_functionDecl, cur_this_expr))
      return true;
  }
  else
  {
    const auto &ctor_json = find_constructor_ref(cname);
    if (get_func_decl_this_ref(ctor_json, cur_this_expr))
      return true;
  }

  locationt l;
  get_location_from_node(expr, l);
  std::unordered_set<std::string> cname_set = structureTypingMap[_cname];
  assert(!cname_set.empty());

  exprt balance;
  bool is_call_w_options = is_func_call && options.is_array();
  if (is_call_w_options)
  {
    // this can be value, gas ...
    // For now, we only consider value
    nlohmann::json literal_type = {
      {"typeIdentifier", "t_uint256"}, {"typeString", "uint256"}};
    if (get_expr(options[0], literal_type, balance))
      return true;
  }

  if (cname_set.size() == 1)
  {
    // skip the "if(..)"
    if (is_func_call)
    {
      // wrap it
      exprt front_block = code_blockt();
      exprt back_block = code_blockt();
      if (is_call_w_options)
      {
        if (model_transaction(
              expr, cur_this_expr, base, balance, l, front_block, back_block))
        {
          log_error("failed to model the transaction property changes");
          return true;
        }
      }
      else if (get_high_level_call_wrapper(
                 cname, cur_this_expr, front_block, back_block))
        return true;

      for (auto op : front_block.operands())
        move_to_front_block(op);
      for (auto op : back_block.operands())
        move_to_back_block(op);
    }

    return false; // since it has only one possible option, no need to futher binding
  }

  // now we need to consider the binding

  if (member.type().get("#sol_type") == "TUPLE_RETURNS")
  {
    log_error("Unsupported return tuple");
    return true;
  }

  bool is_return_void = member.type().is_empty() ||
                        (member.type().is_code() &&
                         to_code_type(member.type()).return_type().is_empty());

  // construct auxilirary funciton
  // e.g.
  //  Bank target;
  //  target.withdraw()
  // => Bank_withdraw(this, this->target)
  assert(!_cname.empty());
  assert(!member.name().empty());
  std::string fname = _cname + "_" + member.name().as_string();
  std::string fid = "sol:@C@" + cname + "@F@" + fname + "#";
  code_typet ft;
  if (!is_return_void)
  {
    if (is_func_call)
      ft.return_type() = to_code_type(member.type()).return_type();
    else
      ft.return_type() = member.type();
  }
  else
    ft.return_type() = empty_typet();
  symbolt fs;
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(fs, debug_modulename, ft, fname, fid, locationt());
  fs.lvalue = true;
  fs.is_extern = false;
  fs.file_local = true;
  auto &added_fsymbol = *move_symbol_to_context(fs);

  // add this pointer to arguments
  get_function_this_pointer_param(
    cname, fid, debug_modulename, locationt(), ft);
  // add base to arguments
  code_typet::argumentt base_param;
  std::string base_name = "base";
  std::string base_id =
    "sol:@C@" + cname + "@F@" + fname + "@" + base_name + "#";
  base_param.cmt_base_name(base_name);
  base_param.cmt_identifier(base_id);

  base_param.type() = base.type();
  symbolt param_symbol;
  get_default_symbol(
    param_symbol,
    debug_modulename,
    base_param.type(),
    base_name,
    base_id,
    locationt());
  param_symbol.lvalue = true;
  param_symbol.is_parameter = true;
  param_symbol.file_local = true;
  if (context.find_symbol(base_id) == nullptr)
  {
    context.move_symbol_to_context(param_symbol);
  }

  ft.arguments().push_back(base_param);
  exprt new_base = symbol_expr(*context.find_symbol(base_id));

  added_fsymbol.type = ft;
  //! we need to move it to the struct symbol
  // this is because we use the member from the contract
  move_builtin_to_contract(cname, symbol_expr(added_fsymbol), true);

  exprt this_expr;
  if (get_func_decl_this_ref(cname, fid, this_expr))
    return true;

  // function body

  // add esbmc_hide label
  exprt func_body = code_blockt();
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.move_to_operands(label);

  // get 'x._ESBMC_bind_cname'
  exprt bind_expr = member_exprt(
    new_base, "_ESBMC_bind_cname", pointer_typet(signed_char_type()));

  // get memebr type
  exprt tmp;
  if (!is_return_void)
  {
    std::string aux_name, aux_id;
    aux_name =
      "$return_" + base.name().as_string() + "_" + member.name().as_string();
    aux_id = "sol:@" + aux_name + std::to_string(aux_counter++);
    symbolt s;
    typet t = ft.return_type();
    if (t.id() == irept::id_code)
      t = to_code_type(t).return_type();
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    get_default_symbol(
      s, debug_modulename, t, aux_name, aux_id, member.location());
    auto &added_symbol = *move_symbol_to_context(s);
    s.lvalue = true;
    s.file_local = true;
    code_declt decl(symbol_expr(added_symbol));

    tmp = symbol_expr(added_symbol);
    func_body.move_to_operands(decl);
  }

  // rhs
  // @str: contract name
  for (auto str : cname_set)
  {
    // strcmp（_ESBMC_NODET_cont_name, Base)
    exprt cname_string;
    typet ct = pointer_typet(signed_char_type());
    ct.cmt_constant(true);
    get_symbol_decl_ref(str, "sol:@" + str, ct, cname_string);

    // since we do not modify the string, and it always point to the known object
    exprt _cmp_cname = exprt("=", pointer_typet(signed_char_type()));
    _cmp_cname.operands().push_back(bind_expr);
    _cmp_cname.operands().push_back(cname_string);

    // member access
    exprt memcall;
    exprt rhs;

    exprt _base;
    get_static_contract_instance_ref(str, _base);

    // ?fix address?. e.g.
    // B target = B(_addr); // previously
    // base->$address =  _ESBMC_Object_B.$address // note that pointer this->target == base

    bool is_revert = false;
    if (is_func_call)
    {
      // e.g. x.call() y.call(). we need to find the definiton of the call beyond the contract x/y seperately
      // get call
      std::string func_name = member.name().as_string();
      assert(!func_name.empty());
      const nlohmann::json &member_decl_ref = get_func_decl_ref(str, func_name);
      if (member_decl_ref == empty_json)
        continue;

      exprt comp;
      if (get_func_decl_ref(member_decl_ref, comp))
        return true;

      side_effect_expr_function_callt call;
      if (get_non_library_function_call(member_decl_ref, expr, call))
        return true;

      // func(&this) => func(&_ESBMC_Object_str)
      call.arguments().at(0) = _base;
      memcall = call;
    }
    else
    {
      assert(!member.name().empty());

      if (inheritanceMap[_cname].count(str))
        memcall = member_exprt(_base, member.name(), member.type());
      else
      {
        // check if the state variable exsist in the target contract
        // signature: type + name
        // this is due to that the structureTypeMap only ensure the function signature matched
        if (is_var_getter_matched(
              str, member.name().as_string(), member.type()))
          memcall = member_exprt(_base, member.name(), member.type());
        else
        {
          // this should be a revert
          // however, esbmc-kind havs trouble in __ESBMC_asusme(false) (v7.8)
          side_effect_expr_function_callt call;
          get_library_function_call_no_args(
            "__ESBMC_assume", "c:@F@__ESBMC_assume", empty_typet(), l, call);

          exprt arg = false_exprt();
          call.arguments().push_back(arg);
          memcall = call;
          is_revert = true;
        }
      }
    }
    rhs = memcall;
    if (!is_return_void && !is_revert)
    {
      exprt _assign = side_effect_exprt("assign", tmp.type());
      convert_type_expr(ns, memcall, tmp.type());
      _assign.copy_to_operands(tmp, memcall);
      rhs = _assign;
    }
    convert_expression_to_code(rhs);

    // wrap it
    if (is_func_call)
    {
      exprt front_block = code_blockt();
      exprt back_block = code_blockt();
      if (is_call_w_options)
      {
        if (model_transaction(
              expr, this_expr, new_base, balance, l, front_block, back_block))
        {
          log_error("failed to model the transaction property changes");
          return true;
        }
      }
      else if (get_high_level_call_wrapper(
                 cname, this_expr, front_block, back_block))
        return true;

      // if-body
      code_blockt block;
      for (auto &op : front_block.operands())
        block.move_to_operands(op);
      block.move_to_operands(rhs);
      for (auto &op : back_block.operands())
        block.move_to_operands(op);
      rhs = block;
    }
    else
    {
      code_blockt block;
      block.move_to_operands(rhs);
      rhs = block;
    }

    codet if_expr("ifthenelse");
    if_expr.move_to_operands(_cmp_cname, rhs);
    if_expr.location() = l;
    //? empty file?
    if_expr.location().file("");
    func_body.move_to_operands(if_expr);
  }

  // return
  if (!is_return_void)
  {
    code_returnt _ret;
    _ret.return_value() = tmp;
    func_body.move_to_operands(_ret);
  }

  added_fsymbol.value = func_body;

  // construct function call
  side_effect_expr_function_callt _call;
  _call.function() = symbol_expr(added_fsymbol);
  _call.type() = ft.return_type();
  _call.location() = l;
  // bank_withdraw(this, this->target)
  _call.arguments().push_back(cur_this_expr);
  _call.arguments().push_back(base);

  new_expr = _call;

  log_debug("solidity", "\tSuccessfully modelled member access.");
  return false;
}

/** e.g.
 * x.call{value: val}("")
 * @base: (this->)x
 * @mem_name: call
 * @options: val
 * @arg: ""
 */
bool solidity_convertert::get_low_level_member_accsss(
  const nlohmann::json &expr,
  const nlohmann::json &options,
  const std::string mem_name,
  const exprt &base,
  const exprt &arg,
  exprt &new_expr)
{
  log_debug("solidity", "Getting low-level member access");

  locationt loc;
  get_location_from_node(expr, loc);
  side_effect_expr_function_callt call;

  std::string cname;
  get_current_contract_name(expr, cname);
  if (cname.empty())
    return true;

  // get this
  exprt this_object;
  if (current_functionDecl)
  {
    if (get_func_decl_this_ref(*current_functionDecl, this_object))
      return true;
  }
  else if (!expr.empty())
  {
    if (get_ctor_decl_this_ref(expr, this_object))
      return true;
  }
  else
  {
    log_error("cannot get this object ref");
    return true;
  }

  if (mem_name == "call")
  {
    std::string func_name = "call";
    exprt addr = base;
    if (options != nullptr)
    {
      // do call#1(this, addr, value) (call with ether)
      addr.type().set("#sol_type", "ADDRESS_PAYABLE");
      exprt value;
      // type should be uint256
      nlohmann::json literal_type = {
        {"typeIdentifier", "t_uint256"}, {"typeString", "uint256"}};

      if (get_expr(options[0], literal_type, value))
        return true;

      std::string func_id = "sol:@C@" + cname + "@F@$call#1";

      get_library_function_call_no_args(
        func_name, func_id, bool_type(), loc, call);
      call.arguments().push_back(this_object);
      call.arguments().push_back(addr);
      call.arguments().push_back(value);
    }
    else
    {
      // To call#0(this, addr)
      addr.type().set("#sol_type", "ADDRESS");

      std::string func_id = "sol:@C@" + cname + "@F@$call#0";
      get_library_function_call_no_args(
        func_name, func_id, bool_type(), loc, call);
      call.arguments().push_back(this_object);
      call.arguments().push_back(addr);
    }

    // convert the return value to tuple
    convert_expression_to_code(call);
    move_to_front_block(call);

    symbolt dump;
    get_llc_ret_tuple(dump);
    dump.value.op0() = call;
    new_expr = symbol_expr(dump);
  }
  else if (mem_name == "transfer")
  {
    // transfer(this, to_addr, balance_value)
    exprt addr = base;
    assert(!arg.is_nil());

    std::string func_name = "transfer";
    std::string func_id = "sol:@C@" + cname + "@F@$transfer#0";
    get_library_function_call_no_args(
      func_name, func_id, bool_type(), loc, call);
    call.arguments().push_back(this_object);
    call.arguments().push_back(addr);
    call.arguments().push_back(arg);

    new_expr = call;
  }
  else if (mem_name == "send")
  {
    // send(this, to_addr, balance_value)
    exprt addr = base;
    assert(!arg.is_nil());

    std::string func_name = "send";
    std::string func_id = "sol:@C@" + cname + "@F@$send#0";
    get_library_function_call_no_args(
      func_name, func_id, bool_type(), loc, call);
    call.arguments().push_back(this_object);
    call.arguments().push_back(addr);
    call.arguments().push_back(arg);

    new_expr = call;
  }
  else
  {
    log_error("unsupported low-level call type {}", mem_name);
    return true;
  }

  return false;
}

void solidity_convertert::get_bind_cname_func_name(
  const std::string &cname,
  std::string &fname,
  std::string &fid)
{
  fname = "initialize_" + cname + +"_bind_cname";
  fid = "sol:@F@" + fname + "#";
}

// return expr: contract_instance._ESBMC_bind_cname
bool solidity_convertert::get_bind_cname_expr(
  const nlohmann::json &json,
  exprt &bind_cname_expr)
{
  const nlohmann::json &parent = find_last_parent(src_ast_json, json);
  locationt l;
  get_location_from_node(json, l);
  exprt lvar;

  if (!parent.contains("nodeType"))
  {
    log_error("{}", parent.dump());
    abort();
  }
  if (parent["nodeType"] == "ExpressionStatement")
    return true; // e.g. new Base(); Base(_addr); with no lvalue
  else if (parent["nodeType"] == "VariableDeclarationStatement")
  {
    assert(parent.contains("declarations"));
    if (get_var_decl_ref(parent["declarations"][0], true, lvar))
      return true;
  }
  else if (parent["nodeType"] == "VariableDeclaration")
  {
    if (get_var_decl_ref(parent, true, lvar))
      return true;
  }
  else if (parent["nodeType"] == "Assignment")
  {
    if (get_expr(parent["leftHandSide"], lvar))
      return true;
  }
  else
  {
    log_warning(
      "got Unexpected nodeType: {}", parent["nodeType"].get<std::string>());
    return true;
  }

  bind_cname_expr =
    member_exprt(lvar, "_ESBMC_bind_cname", pointer_typet(signed_char_type()));
  bind_cname_expr.location() = l;
  return false;
}

/**
 * symbol
 *   * identifier: tag-Bank
*/
void solidity_convertert::get_new_object(const typet &t, exprt &this_object)
{
  log_debug("solidity", "\t\tget this object ref");
  assert(t.is_symbol());

  exprt temporary = exprt("new_object");
  temporary.set("#lvalue", true);
  temporary.type() = t;
  this_object = temporary;
}

// add `call(address _addr)` to the contract
// If it contains the funciton signature, it should be directly converted to the function calls rathe than invoke this `call`
// e.g. addr.call(abi.encodeWithSignature("doSomething(uint256)", 123))
// => _ESBMC_Object_Base.doSomething(123);
bool solidity_convertert::get_call_definition(
  const std::string &cname,
  exprt &new_expr)
{
  std::string call_name = "call";
  std::string call_id = "sol:@C@" + cname + "@F@$call#0";
  symbolt s;
  // the return value should be (bool, string)
  // however, we cannot handle the string, therefore we only return bool
  // and make it (x.call(), nondet_uint_expr)
  code_typet t;
  t.return_type() = bool_type();
  t.return_type().set("#sol_type", "BOOL");
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(s, debug_modulename, t, call_name, call_id, locationt());
  auto &added_symbol = *move_symbol_to_context(s);
  get_function_this_pointer_param(
    cname, call_id, debug_modulename, locationt(), t);

  // param: address _addr;
  std::string addr_name = "_addr";
  std::string addr_id = "sol:@C@" + cname + "@F@call@" + addr_name + "#" +
                        std::to_string(aux_counter++);
  typet addr_t = unsignedbv_typet(160);
  addr_t.set("#sol_type", "ADDRESS");
  symbolt addr_s;
  get_default_symbol(
    addr_s, debug_modulename, addr_t, addr_name, addr_id, locationt());
  auto addr_added_symbol = *move_symbol_to_context(addr_s);

  code_typet::argumentt param = code_typet::argumentt();
  param.type() = addr_t;
  param.cmt_base_name(addr_name);
  param.cmt_identifier(addr_id);
  t.arguments().push_back(param);

  added_symbol.type = t;

  // body:
  /*
  if(_addr == _ESBMC_Object_x) 
  {
    *Also check if it has public or external non-ctor function
    old_sender = msg_sender
    meg_sender = this.address
    _ESBMC_Nondet_Extcall_x();
    msg_sender = old_sender
    return true;
  }
  if(...) {...}
  
  return false;
  */
  code_blockt func_body;

  // add __ESBMC_HIDE label
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.move_to_operands(label);

  exprt addr_expr = symbol_expr(addr_added_symbol);
  exprt msg_sender = symbol_expr(*context.find_symbol("c:@msg_sender"));
  symbolt this_sym = *context.find_symbol(call_id + "#this");
  exprt this_expr = symbol_expr(this_sym);
  exprt this_address = member_exprt(this_expr, "$address", addr_t);

  // uint160_t old_sender =  msg_sender;
  typet _addr_t = unsignedbv_typet(160);
  _addr_t.set("#sol_type", "ADDRESS");
  symbolt old_sender;
  get_default_symbol(
    old_sender,
    debug_modulename,
    _addr_t,
    "old_sender",
    "sol:@C@" + cname + "@F@old_sender#" + std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_sender = *move_symbol_to_context(old_sender);
  code_declt old_sender_decl(symbol_expr(added_old_sender));
  added_old_sender.value = msg_sender;
  old_sender_decl.operands().push_back(msg_sender);
  func_body.move_to_operands(old_sender_decl);

  for (auto str : contractNamesList)
  {
    if (!has_callable_func(str))
      continue;

    code_blockt then;

    // msg_sender = this.address;
    exprt assign_sender = side_effect_exprt("assign", addr_t);
    assign_sender.copy_to_operands(msg_sender, this_address);
    convert_expression_to_code(assign_sender);
    then.move_to_operands(assign_sender);

    if (is_reentry_check)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = true;
      exprt assign_lock = side_effect_exprt("assign", bool_type());
      assign_lock.copy_to_operands(_mutex, true_exprt());
      convert_expression_to_code(assign_lock);
      then.move_to_operands(assign_lock);
    }

    // _ESBMC_Nondet_Extcall_x();
    code_function_callt call;
    if (get_unbound_funccall(str, call))
      return true;
    then.move_to_operands(call);

    if (is_reentry_check)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = false;
      exprt assign_unlock = side_effect_exprt("assign", bool_type());
      assign_unlock.copy_to_operands(_mutex, false_exprt());
      convert_expression_to_code(assign_unlock);
      then.move_to_operands(assign_unlock);
    }

    // msg_sender = old_sender;
    exprt assign_sender_restore = side_effect_exprt("assign", addr_t);
    assign_sender_restore.copy_to_operands(
      msg_sender, symbol_expr(added_old_sender));
    convert_expression_to_code(assign_sender_restore);
    then.move_to_operands(assign_sender_restore);

    // return true;
    code_returnt ret_true;
    ret_true.return_value() = true_exprt();
    then.move_to_operands(ret_true);

    // _addr == _ESBMC_Object_str.$address
    exprt static_ins;
    get_static_contract_instance_ref(str, static_ins);
    exprt mem_addr = member_exprt(static_ins, "$address", addr_t);
    exprt _equal = exprt("=", bool_type());
    _equal.operands().push_back(addr_expr);
    _equal.operands().push_back(mem_addr);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, then);
    func_body.move_to_operands(if_expr);
  }

  // add "Return false;" in the end
  code_returnt return_expr;
  return_expr.return_value() = false_exprt();
  func_body.move_to_operands(return_expr);

  added_symbol.value = func_body;
  new_expr = symbol_expr(added_symbol);
  return false;
}

/** e.g. x = target.deposit{value: msg.value}()
 * @expr: member_call json
 * @this_expr: this->(target)
 * @base: target
 * @value: msg.value
 * @block: returns
*/
bool solidity_convertert::model_transaction(
  const nlohmann::json &expr,
  const exprt &this_expr,
  const exprt &base,
  const exprt &value,
  const locationt &loc,
  exprt &front_block,
  exprt &back_block)
{
  log_debug("solidity", "modelling the transaction property changes");
  /*
  old_sender = msg.sender
  old_value = msg.value
  msg.sender = instance.$address
  msg.value = instance.$balance
  instance.$balance -= value
  base.$balance += value
  (_ESBMC_mutext_Base = true)
  (call to payable func)
  (_ESBMC_mutext_Base = false)
  msg.sender = old_sender;
  msg.value = old_value
  */
  front_block = code_blockt();
  back_block = code_blockt();
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  std::string cname;
  get_current_contract_name(expr, cname);
  if (cname.empty())
    return true;

  typet val_t = unsignedbv_typet(256);
  exprt msg_value = symbol_expr(*context.find_symbol("c:@msg_value"));

  if (get_high_level_call_wrapper(cname, this_expr, front_block, back_block))
    return true;

  exprt this_balance = member_exprt(this_expr, "$balance", val_t);

  symbolt old_value;
  get_default_symbol(
    old_value,
    debug_modulename,
    unsignedbv_typet(256),
    "old_value",
    "sol:@old_value#" + std::to_string(aux_counter++),
    loc);
  symbolt &added_old_value = *move_symbol_to_context(old_value);
  code_declt old_val_decl(symbol_expr(added_old_value));
  added_old_value.value = msg_value;
  old_val_decl.operands().push_back(msg_value);
  front_block.move_to_operands(old_val_decl);

  // msg_value = _val;
  exprt assign_val = side_effect_exprt("assign", value.type());
  assign_val.copy_to_operands(msg_value, value);
  convert_expression_to_code(assign_val);
  front_block.move_to_operands(assign_val);

  // if(this.balance < val) return false;
  exprt less_than = exprt("<", bool_type());
  less_than.copy_to_operands(this_balance, value);
  codet cmp_less_than("ifthenelse");
  code_returnt ret_false;
  ret_false.return_value() = false_exprt();
  cmp_less_than.copy_to_operands(less_than, ret_false);
  front_block.move_to_operands(cmp_less_than);

  // this.balance -= _val;
  exprt sub_assign = side_effect_exprt("assign-", val_t);
  sub_assign.copy_to_operands(this_balance, value);
  convert_expression_to_code(sub_assign);
  front_block.move_to_operands(sub_assign);

  // base.balance += _val;
  exprt target_balance = member_exprt(base, "$balance", val_t);
  exprt add_assign = side_effect_exprt("assign+", val_t);
  add_assign.copy_to_operands(target_balance, value);
  convert_expression_to_code(add_assign);
  front_block.move_to_operands(add_assign);

  // msg_value = old_value;
  exprt assign_val_restore = side_effect_exprt("assign", value.type());
  assign_val_restore.copy_to_operands(msg_value, symbol_expr(added_old_value));
  convert_expression_to_code(assign_val_restore);
  back_block.move_to_operands(assign_val_restore);

  convert_expression_to_code(front_block);
  convert_expression_to_code(back_block);
  return false;
}

// `call(address _addr, uint _val)`
bool solidity_convertert::get_call_value_definition(
  const std::string &cname,
  exprt &new_expr)
{
  std::string call_name = "call";
  std::string call_id = "sol:@C@" + cname + "@F@$call#1";
  symbolt s;
  // the return value should be (bool, string)
  // however, we cannot handle the string, therefore we only return bool
  // and make it (x.call(), nondet_uint_expr) later
  code_typet t;
  t.return_type() = bool_type();
  t.return_type().set("#sol_type", "BOOL");
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(s, debug_modulename, t, call_name, call_id, locationt());
  auto &added_symbol = *move_symbol_to_context(s);
  get_function_this_pointer_param(
    cname, call_id, debug_modulename, locationt(), t);

  // param: address _addr;
  std::string addr_name = "_addr";
  std::string addr_id = "sol:@C@" + cname + "@F@call@" + addr_name + "#1";
  typet addr_t = unsignedbv_typet(160);
  addr_t.set("#sol_type", "ADDRESS_PAYABLE");
  symbolt addr_s;
  get_default_symbol(
    addr_s, debug_modulename, addr_t, addr_name, addr_id, locationt());
  auto addr_added_symbol = *move_symbol_to_context(addr_s);

  code_typet::argumentt param = code_typet::argumentt();
  param.type() = addr_t;
  param.cmt_base_name(addr_name);
  param.cmt_identifier(addr_id);
  t.arguments().push_back(param);

  // param: uint _val;
  std::string val_name = "_val";
  std::string val_id = "sol:@C@" + cname + "@F@call@" + val_name + "#1";
  typet val_t = unsignedbv_typet(256);
  symbolt val_s;
  get_default_symbol(
    val_s, debug_modulename, val_t, val_name, val_id, locationt());
  auto val_added_symbol = *move_symbol_to_context(val_s);

  param = code_typet::argumentt();
  param.type() = val_t;
  param.cmt_base_name(val_name);
  param.cmt_identifier(val_id);
  t.arguments().push_back(param);

  added_symbol.type = t;

  // body:
  /*
  __ESBMC_Hide;
  uint256_t old_value = msg_value;
  uint160_t old_sender =  msg_sender;
  if(_addr == _ESBMC_Object_x.$address) 
  {    
    *! we do not consider gas consumption

    msg_value = value 
    msg_sender = this.address;
    if(this.balance < x)      <-- simulate EVM rollback
      return false;
    this.balance -= x; 
    _ESBMC_Object_x.balance += x; 

    _ESBMC_Object_x.receive() * or fallback

    msg_value = old_value;
    msg_sender = old_sender
    return true;
  }
  if(...) {...}
  
  return false;
  */
  code_blockt func_body;
  exprt addr_expr = symbol_expr(addr_added_symbol);
  exprt val_expr = symbol_expr(val_added_symbol);

  // add __ESBMC_HIDE label
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  exprt msg_sender = symbol_expr(*context.find_symbol("c:@msg_sender"));
  exprt msg_value = symbol_expr(*context.find_symbol("c:@msg_value"));
  symbolt this_sym = *context.find_symbol(call_id + "#this");
  exprt this_expr = symbol_expr(this_sym);
  exprt this_address = member_exprt(this_expr, "$address", addr_t);
  exprt this_balance = member_exprt(this_expr, "$balance", val_t);

  // uint256_t old_value = msg_value;
  symbolt old_value;
  get_default_symbol(
    old_value,
    debug_modulename,
    unsignedbv_typet(256),
    "old_value",
    "sol:@C@" + cname + "@F@call@old_value#" + std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_value = *move_symbol_to_context(old_value);
  code_declt old_val_decl(symbol_expr(added_old_value));
  added_old_value.value = msg_value;
  old_val_decl.operands().push_back(msg_value);
  func_body.move_to_operands(old_val_decl);

  // uint160_t old_sender =  msg_sender;
  typet _addr_t = unsignedbv_typet(160);
  _addr_t.set("#sol_type", "ADDRESS");
  symbolt old_sender;
  get_default_symbol(
    old_sender,
    debug_modulename,
    _addr_t,
    "old_sender",
    "sol:@C@" + cname + "@F@call@old_sender#" + std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_sender = *move_symbol_to_context(old_sender);
  code_declt old_sender_decl(symbol_expr(added_old_sender));
  added_old_sender.value = msg_sender;
  old_sender_decl.operands().push_back(msg_sender);
  func_body.move_to_operands(old_sender_decl);

  for (auto str : contractNamesList)
  {
    // Here, we only consider if there is receive and fallback function
    // as the call with signature should be directly modelled.
    // order:
    // 1. match payable receive
    // 2. match payable fallback
    // 3. return false (revert)
    nlohmann::json decl_ref;
    if (has_target_function(str, "receive"))
      decl_ref = get_func_decl_ref(str, "receive");
    else if (has_target_function(str, "fallback"))
      decl_ref = get_func_decl_ref(str, "fallback");
    else
      // other payable function
      continue;
    if (decl_ref["stateMutability"] != "payable")
      continue;

    code_blockt then;

    // _addr == _ESBMC_Object_str.$address
    exprt static_ins;
    get_static_contract_instance_ref(str, static_ins);
    exprt mem_addr = member_exprt(static_ins, "$address", addr_t);

    exprt _equal = exprt("=", bool_type());
    _equal.operands().push_back(addr_expr);
    _equal.operands().push_back(mem_addr);

    // msg_value = _val;
    exprt assign_val = side_effect_exprt("assign", val_expr.type());
    assign_val.copy_to_operands(msg_value, val_expr);
    convert_expression_to_code(assign_val);
    then.move_to_operands(assign_val);

    // msg_sender = this.$address;
    exprt assign_sender = side_effect_exprt("assign", addr_t);
    assign_sender.copy_to_operands(msg_sender, this_address);
    convert_expression_to_code(assign_sender);
    then.move_to_operands(assign_sender);

    // if(this.balance < val) return false;
    exprt less_than = exprt("<", bool_type());
    less_than.copy_to_operands(this_balance, val_expr);
    codet cmp_less_than("ifthenelse");
    code_returnt ret_false;
    ret_false.return_value() = false_exprt();
    cmp_less_than.copy_to_operands(less_than, ret_false);
    then.move_to_operands(cmp_less_than);

    // this.balance -= _val;
    exprt sub_assign = side_effect_exprt("assign-", val_t);
    sub_assign.copy_to_operands(this_balance, val_expr);
    convert_expression_to_code(sub_assign);
    then.move_to_operands(sub_assign);

    // _ESBMC_Object_str.balance += _val;
    exprt target_balance = member_exprt(static_ins, "$balance", val_t);
    exprt add_assign = side_effect_exprt("assign+", val_t);
    add_assign.copy_to_operands(target_balance, val_expr);
    convert_expression_to_code(add_assign);
    then.move_to_operands(add_assign);

    if (is_reentry_check)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = true;
      exprt assign_lock = side_effect_exprt("assign", bool_type());
      assign_lock.copy_to_operands(_mutex, true_exprt());
      convert_expression_to_code(assign_lock);
      then.move_to_operands(assign_lock);
    }

    // func_call, e.g. receive(&_ESBMC_Object_str)
    side_effect_expr_function_callt call;
    if (get_non_library_function_call(decl_ref, empty_json, call))
      return true;
    call.arguments().at(0) = static_ins;
    convert_expression_to_code(call);
    then.move_to_operands(call);

    if (is_reentry_check)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = false;
      exprt assign_unlock = side_effect_exprt("assign", bool_type());
      assign_unlock.copy_to_operands(_mutex, false_exprt());
      convert_expression_to_code(assign_unlock);
      then.move_to_operands(assign_unlock);
    }

    // msg_value = old_value;
    exprt assign_val_restore = side_effect_exprt("assign", val_expr.type());
    assign_val_restore.copy_to_operands(
      msg_value, symbol_expr(added_old_value));
    convert_expression_to_code(assign_val_restore);
    then.move_to_operands(assign_val_restore);

    // msg_sender = old_sender;
    exprt assign_sender_restore = side_effect_exprt("assign", addr_t);
    assign_sender_restore.copy_to_operands(
      msg_sender, symbol_expr(added_old_sender));
    convert_expression_to_code(assign_sender_restore);
    then.move_to_operands(assign_sender_restore);

    // return true;
    code_returnt ret_true;
    ret_true.return_value() = true_exprt();
    then.move_to_operands(ret_true);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, then);
    func_body.move_to_operands(if_expr);
  }
  // add "Return false;" in the end
  code_returnt return_expr;
  return_expr.return_value() = false_exprt();
  func_body.move_to_operands(return_expr);

  added_symbol.value = func_body;
  new_expr = symbol_expr(added_symbol);
  return false;
}

bool solidity_convertert::get_transfer_definition(
  const std::string &cname,
  exprt &new_expr)
{
  std::string call_name = "transfer";
  std::string call_id = "sol:@C@" + cname + "@F@$transfer#0";
  code_typet t;
  t.return_type() = bool_type();
  t.return_type().set("#sol_type", "BOOL");
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  symbolt s;
  get_default_symbol(s, debug_modulename, t, call_name, call_id, locationt());
  auto &added_symbol = *move_symbol_to_context(s);
  get_function_this_pointer_param(
    cname, call_id, debug_modulename, locationt(), t);

  // param: address _addr;
  std::string addr_name = "_addr";
  std::string addr_id = "sol:@C@" + cname + "@F@transfer@" + addr_name + "#0";
  typet addr_t = unsignedbv_typet(160);
  addr_t.set("#sol_type", "ADDRESS_PAYABLE");
  symbolt addr_s;
  get_default_symbol(
    addr_s, debug_modulename, addr_t, addr_name, addr_id, locationt());
  auto addr_added_symbol = *move_symbol_to_context(addr_s);

  code_typet::argumentt param = code_typet::argumentt();
  param.type() = addr_t;
  param.cmt_base_name(addr_name);
  param.cmt_identifier(addr_id);
  t.arguments().push_back(param);

  // param: uint _val;
  std::string val_name = "_val";
  std::string val_id = "sol:@C@" + cname + "@F@transfer@" + val_name + "#0";
  typet val_t = unsignedbv_typet(256);
  symbolt val_s;
  get_default_symbol(
    val_s, debug_modulename, val_t, val_name, val_id, locationt());
  auto val_added_symbol = *move_symbol_to_context(val_s);

  param = code_typet::argumentt();
  param.type() = val_t;
  param.cmt_base_name(val_name);
  param.cmt_identifier(val_id);
  t.arguments().push_back(param);

  added_symbol.type = t;

  code_blockt func_body;
  exprt addr_expr = symbol_expr(addr_added_symbol);
  exprt val_expr = symbol_expr(val_added_symbol);

  // add __ESBMC_HIDE label
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  exprt msg_sender = symbol_expr(*context.find_symbol("c:@msg_sender"));
  exprt msg_value = symbol_expr(*context.find_symbol("c:@msg_value"));
  symbolt this_sym = *context.find_symbol(call_id + "#this");
  exprt this_expr = symbol_expr(this_sym);
  exprt this_address = member_exprt(this_expr, "$address", addr_t);
  exprt this_balance = member_exprt(this_expr, "$balance", val_t);

  // uint256_t old_value = msg_value;
  symbolt old_value;
  get_default_symbol(
    old_value,
    debug_modulename,
    unsignedbv_typet(256),
    "old_value",
    "sol:@C@" + cname + "@F@transfer@old_value#" +
      std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_value = *move_symbol_to_context(old_value);
  code_declt old_val_decl(symbol_expr(added_old_value));
  added_old_value.value = msg_value;
  old_val_decl.operands().push_back(msg_value);
  func_body.move_to_operands(old_val_decl);

  // uint160_t old_sender =  msg_sender;
  typet _addr_t = unsignedbv_typet(160);
  _addr_t.set("#sol_type", "ADDRESS");
  symbolt old_sender;
  get_default_symbol(
    old_sender,
    debug_modulename,
    _addr_t,
    "old_sender",
    "sol:@C@" + cname + "@F@transfer@old_sender#" +
      std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_sender = *move_symbol_to_context(old_sender);
  code_declt old_sender_decl(symbol_expr(added_old_sender));
  added_old_sender.value = msg_sender;
  old_sender_decl.operands().push_back(msg_sender);
  func_body.move_to_operands(old_sender_decl);

  for (auto str : contractNamesList)
  {
    // Here, we only consider if there is receive and fallback function
    // as the call with signature should be directly modelled.
    // order:
    // 1. match payable receive
    // 2. match payable fallback
    // 3. return false (revert)
    nlohmann::json decl_ref;
    if (has_target_function(str, "receive"))
      decl_ref = get_func_decl_ref(str, "receive");
    else if (has_target_function(str, "fallback"))
      decl_ref = get_func_decl_ref(str, "fallback");
    else
      // other payable function
      continue;
    if (decl_ref["stateMutability"] != "payable")
      continue;

    code_blockt then;

    // _addr == _ESBMC_Object_str.$address
    exprt static_ins;
    get_static_contract_instance_ref(str, static_ins);
    exprt mem_addr = member_exprt(static_ins, "$address", addr_t);

    exprt _equal = exprt("=", bool_type());
    _equal.operands().push_back(addr_expr);
    _equal.operands().push_back(mem_addr);

    // msg_value = _val;
    exprt assign_val = side_effect_exprt("assign", val_expr.type());
    assign_val.copy_to_operands(msg_value, val_expr);
    convert_expression_to_code(assign_val);
    then.move_to_operands(assign_val);

    // msg_sender = this.$address;
    exprt assign_sender = side_effect_exprt("assign", addr_t);
    assign_sender.copy_to_operands(msg_sender, this_address);
    convert_expression_to_code(assign_sender);
    then.move_to_operands(assign_sender);

    // if(this.balance < val) return false;
    exprt less_than = exprt("<", bool_type());
    less_than.copy_to_operands(this_balance, val_expr);
    codet cmp_less_than("ifthenelse");
    code_returnt ret_false;
    ret_false.return_value() = false_exprt();
    cmp_less_than.copy_to_operands(less_than, ret_false);
    then.move_to_operands(cmp_less_than);

    // this.balance -= _val;
    exprt sub_assign = side_effect_exprt("assign-", val_t);
    sub_assign.copy_to_operands(this_balance, val_expr);
    convert_expression_to_code(sub_assign);
    then.move_to_operands(sub_assign);

    // _ESBMC_Object_str.balance += _val;
    exprt target_balance = member_exprt(static_ins, "$balance", val_t);
    exprt add_assign = side_effect_exprt("assign+", val_t);
    add_assign.copy_to_operands(target_balance, val_expr);
    convert_expression_to_code(add_assign);
    then.move_to_operands(add_assign);

    if (is_reentry_check)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = true;
      exprt assign_lock = side_effect_exprt("assign", bool_type());
      //! Do not use gen_one(bool_type()) to replace true_exprt()
      //! it will make the verification process stuck somehow
      assign_lock.copy_to_operands(_mutex, true_exprt());
      convert_expression_to_code(assign_lock);
      then.move_to_operands(assign_lock);
    }

    // func_call, e.g. receive(&_ESBMC_Object_str)
    side_effect_expr_function_callt call;
    if (get_non_library_function_call(decl_ref, empty_json, call))
      return true;
    call.arguments().at(0) = static_ins;
    convert_expression_to_code(call);
    then.move_to_operands(call);

    if (is_reentry_check)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = false;
      exprt assign_unlock = side_effect_exprt("assign", bool_type());
      assign_unlock.copy_to_operands(_mutex, false_exprt());
      convert_expression_to_code(assign_unlock);
      then.move_to_operands(assign_unlock);
    }

    // msg_value = old_value;
    exprt assign_val_restore = side_effect_exprt("assign", val_expr.type());
    assign_val_restore.copy_to_operands(
      msg_value, symbol_expr(added_old_value));
    convert_expression_to_code(assign_val_restore);
    then.move_to_operands(assign_val_restore);

    // msg_sender = old_sender;
    exprt assign_sender_restore = side_effect_exprt("assign", addr_t);
    assign_sender_restore.copy_to_operands(
      msg_sender, symbol_expr(added_old_sender));
    convert_expression_to_code(assign_sender_restore);
    then.move_to_operands(assign_sender_restore);

    // return true;
    code_returnt ret_true;
    ret_true.return_value() = true_exprt();
    then.move_to_operands(ret_true);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, then);
    func_body.move_to_operands(if_expr);
  }
  // add "Return false;" in the end
  code_returnt return_expr;
  return_expr.return_value() = false_exprt();
  func_body.move_to_operands(return_expr);

  added_symbol.value = func_body;
  new_expr = symbol_expr(added_symbol);
  return false;
}

bool solidity_convertert::get_send_definition(
  const std::string &cname,
  exprt &new_expr)
{
  std::string call_name = "send";
  std::string call_id = "sol:@C@" + cname + "@F@$send#0";
  code_typet t;
  t.return_type() = bool_type();
  t.return_type().set("#sol_type", "BOOL");
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  symbolt s;
  get_default_symbol(s, debug_modulename, t, call_name, call_id, locationt());
  auto &added_symbol = *move_symbol_to_context(s);
  get_function_this_pointer_param(
    cname, call_id, debug_modulename, locationt(), t);

  // param: address _addr;
  std::string addr_name = "_addr";
  std::string addr_id = "sol:@C@" + cname + "@F@send@" + addr_name + "#0";
  typet addr_t = unsignedbv_typet(160);
  addr_t.set("#sol_type", "ADDRESS_PAYABLE");
  symbolt addr_s;
  get_default_symbol(
    addr_s, debug_modulename, addr_t, addr_name, addr_id, locationt());
  auto addr_added_symbol = *move_symbol_to_context(addr_s);

  code_typet::argumentt param = code_typet::argumentt();
  param.type() = addr_t;
  param.cmt_base_name(addr_name);
  param.cmt_identifier(addr_id);
  t.arguments().push_back(param);

  // param: uint _val;
  std::string val_name = "_val";
  std::string val_id = "sol:@C@" + cname + "@F@send@" + val_name + "#0";
  typet val_t = unsignedbv_typet(256);
  symbolt val_s;
  get_default_symbol(
    val_s, debug_modulename, val_t, val_name, val_id, locationt());
  auto val_added_symbol = *move_symbol_to_context(val_s);

  param = code_typet::argumentt();
  param.type() = val_t;
  param.cmt_base_name(val_name);
  param.cmt_identifier(val_id);
  t.arguments().push_back(param);

  added_symbol.type = t;

  code_blockt func_body;
  exprt addr_expr = symbol_expr(addr_added_symbol);
  exprt val_expr = symbol_expr(val_added_symbol);

  // add __ESBMC_HIDE label
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  exprt msg_sender = symbol_expr(*context.find_symbol("c:@msg_sender"));
  exprt msg_value = symbol_expr(*context.find_symbol("c:@msg_value"));
  symbolt this_sym = *context.find_symbol(call_id + "#this");
  exprt this_expr = symbol_expr(this_sym);
  exprt this_address = member_exprt(this_expr, "$address", addr_t);
  exprt this_balance = member_exprt(this_expr, "$balance", val_t);

  // uint256_t old_value = msg_value;
  symbolt old_value;
  get_default_symbol(
    old_value,
    debug_modulename,
    unsignedbv_typet(256),
    "old_value",
    "sol:@C@" + cname + "@F@send@old_value#" + std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_value = *move_symbol_to_context(old_value);
  code_declt old_val_decl(symbol_expr(added_old_value));
  added_old_value.value = msg_value;
  old_val_decl.operands().push_back(msg_value);
  func_body.move_to_operands(old_val_decl);

  // uint160_t old_sender =  msg_sender;
  typet _addr_t = unsignedbv_typet(160);
  _addr_t.set("#sol_type", "ADDRESS");
  symbolt old_sender;
  get_default_symbol(
    old_sender,
    debug_modulename,
    _addr_t,
    "old_sender",
    "sol:@C@" + cname + "@F@send@old_sender#" + std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_sender = *move_symbol_to_context(old_sender);
  code_declt old_sender_decl(symbol_expr(added_old_sender));
  added_old_sender.value = msg_sender;
  old_sender_decl.operands().push_back(msg_sender);
  func_body.move_to_operands(old_sender_decl);

  for (auto str : contractNamesList)
  {
    // Here, we only consider if there is receive and fallback function
    // as the call with signature should be directly modelled.
    // order:
    // 1. match payable receive
    // 2. match payable fallback
    // 3. return false (revert)
    nlohmann::json decl_ref;
    if (has_target_function(str, "receive"))
      decl_ref = get_func_decl_ref(str, "receive");
    else if (has_target_function(str, "fallback"))
      decl_ref = get_func_decl_ref(str, "fallback");
    else
      // other payable function
      continue;
    if (decl_ref["stateMutability"] != "payable")
      continue;

    code_blockt then;

    // _addr == _ESBMC_Object_str.$address
    exprt static_ins;
    get_static_contract_instance_ref(str, static_ins);
    exprt mem_addr = member_exprt(static_ins, "$address", addr_t);

    exprt _equal = exprt("=", bool_type());
    _equal.operands().push_back(addr_expr);
    _equal.operands().push_back(mem_addr);

    // msg_value = _val;
    exprt assign_val = side_effect_exprt("assign", val_expr.type());
    assign_val.copy_to_operands(msg_value, val_expr);
    convert_expression_to_code(assign_val);
    then.move_to_operands(assign_val);

    // msg_sender = this.$address;
    exprt assign_sender = side_effect_exprt("assign", addr_t);
    assign_sender.copy_to_operands(msg_sender, this_address);
    convert_expression_to_code(assign_sender);
    then.move_to_operands(assign_sender);

    // if(this.balance < val) return false;
    exprt less_than = exprt("<", val_expr.type());
    less_than.copy_to_operands(this_balance, val_expr);
    //! "ifthenelse" has to be declared as codet, not exprt and use convert_expr_to_code
    codet cmp_less_than("ifthenelse");
    code_returnt ret_false;
    ret_false.return_value() = false_exprt();
    cmp_less_than.copy_to_operands(less_than, ret_false);
    then.move_to_operands(cmp_less_than);

    // this.balance -= _val;
    exprt sub_assign = side_effect_exprt("assign-", val_t);
    sub_assign.copy_to_operands(this_balance, val_expr);
    convert_expression_to_code(sub_assign);
    then.move_to_operands(sub_assign);

    // _ESBMC_Object_str.balance += _val;
    exprt target_balance = member_exprt(static_ins, "$balance", val_t);
    exprt add_assign = side_effect_exprt("assign+", val_t);
    add_assign.copy_to_operands(target_balance, val_expr);
    convert_expression_to_code(add_assign);
    then.move_to_operands(add_assign);

    if (is_reentry_check)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = true;
      exprt assign_lock = side_effect_exprt("assign", bool_type());
      assign_lock.copy_to_operands(_mutex, true_exprt());
      convert_expression_to_code(assign_lock);
      then.move_to_operands(assign_lock);
    }

    // func_call, e.g. receive(&_ESBMC_Object_str)
    side_effect_expr_function_callt call;
    if (get_non_library_function_call(decl_ref, empty_json, call))
      return true;
    call.arguments().at(0) = static_ins;
    convert_expression_to_code(call);
    then.move_to_operands(call);

    if (is_reentry_check)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = false;
      exprt assign_unlock = side_effect_exprt("assign", bool_type());
      assign_unlock.copy_to_operands(_mutex, false_exprt());
      convert_expression_to_code(assign_unlock);
      then.move_to_operands(assign_unlock);
    }

    // msg_value = old_value;
    exprt assign_val_restore = side_effect_exprt("assign", val_expr.type());
    assign_val_restore.copy_to_operands(
      msg_value, symbol_expr(added_old_value));
    convert_expression_to_code(assign_val_restore);
    then.move_to_operands(assign_val_restore);

    // msg_sender = old_sender;
    exprt assign_sender_restore = side_effect_exprt("assign", addr_t);
    assign_sender_restore.copy_to_operands(
      msg_sender, symbol_expr(added_old_sender));
    convert_expression_to_code(assign_sender_restore);
    then.move_to_operands(assign_sender_restore);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, then);
    func_body.move_to_operands(if_expr);
  }

  added_symbol.value = func_body;
  new_expr = symbol_expr(added_symbol);

  return false;
}