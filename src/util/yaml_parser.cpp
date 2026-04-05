#include <yaml_parser.h>
#include <util/c_string2expr.h>
#include <util/expr_util.h>

yaml_parser::yaml_parser(
  const std::string &path,
  contextt &ns,
  optionst &options)
  : file_path_(path), context_(ns), options_(options)
{
}

bool yaml_parser::load_file()
{
  try
  {
    root_ = YAML::LoadFile(file_path_);
    if (get_invariants())
      return true;
  }
  catch (const YAML::Exception &e)
  {
    log_error("Failed to parse YAML file '{}': {}", file_path_, e.what());
    return true;
  }

  return false;
}

bool yaml_parser::get_invariants()
{
  if (!root_ || !root_.IsSequence())
    return true;

  for (const auto &entry : root_)
  {
    const auto &content = entry["content"];
    if (!content || !content.IsSequence())
      continue;

    for (const auto &c : content)
    {
      const auto &inv_node = c["invariant"];
      if (!inv_node)
        continue;
      invariant info = parse_invariant(inv_node);
      parsed_invariants_.push_back(std::move(info));
    }
  }

  return false;
}

invariant yaml_parser::parse_invariant(const YAML::Node &node) const
{
  invariant info;
  if (node["type"])
    info.type = type_from_string(node["type"].as<std::string>());
  if (node["value"])
    info.value = node["value"].as<std::string>();
  if (node["format"])
    info.format = node["format"].as<std::string>();

  const auto &loc = node["location"];
  if (loc)
  {
    if (loc["file_name"])
      info.file = loc["file_name"].as<std::string>();
    if (loc["line"])
      info.line = BigInt(loc["line"].as<std::string>().c_str(), 10);
    if (loc["column"])
      info.column = BigInt(loc["column"].as<std::string>().c_str(), 10);
    if (loc["function"])
      info.function = loc["function"].as<std::string>();
  }

  return info;
}

invariant::Type yaml_parser::type_from_string(const std::string &s) const
{
  if (s == "loop_invariant")
    return invariant::loop_invariant;
  if (s == "loop_transition_invariant")
    return invariant::loop_transition_invariant;
  if (s == "location_invariant")
    return invariant::location_invariant;
  if (s == "location_transition_invariant")
    return invariant::location_transition_invariant;

  log_error("Unknown invariant type: {}", s);
  abort();
}

bool yaml_parser::inject_loop_invariants(goto_functionst &goto_functions)
{
  expression_parser parser;
  for (const auto &inv : parsed_invariants_)
  {
    std::string func_id = "c:@F@" + inv.function;
    goto_functionst::function_mapt::iterator m_it =
      goto_functions.function_map.find(func_id);
    if (m_it != goto_functions.function_map.end())
    {
      goto_programt &func = m_it->second.body;
      Forall_goto_program_instructions (it, func)
      {
        int line = std::stoi(it->location.line().as_string());
        if (line == inv.line)
        {
          const expression_node *root = nullptr;
          if (parser.parse(inv.value, root))
          {
            log_warning(
              "failed to build the AST of witness expression: {}, skip it",
              inv.value);
            continue;
          }

          if (options_.get_bool_option("witness-parse-tree"))
          {
            root->dump();
            continue;
          }

          expression_converter converter(context_, it->location);
          exprt expr;
          if (converter.convert(root, expr))
          {
            log_warning(
              "failed to convert the witness AST: {}, skip it", inv.value);
            continue;
          }

          expr2tc guard;
          migrate_expr(expr, guard);

          switch (inv.type)
          {
          case invariant::loop_invariant:
            if (it->is_goto() && !it->is_backwards_goto())
            {
              goto_programt::targett t = func.insert(it);
              t->type = LOOP_INVARIANT;
              t->add_loop_invariant(guard);
              t->location = it->location;
              log_progress(
                "Applied loop invariant: {} in line {}",
                inv.value,
                t->location.line());
            }
            break;

          case invariant::location_invariant:
          {
            goto_programt tmp;
            goto_programt::targett t1 = tmp.add_instruction();
            t1->make_assertion(guard);
            t1->location = it->location;
            func.destructive_insert(it, tmp);

            log_progress(
              "Applied location invariant: {} in line {}",
              inv.value,
              it->location.line());
            break;
          }

          default:
            log_error("unsupported invariant type: {}", inv.value);
            break;
          }

          break;
        }
      }
    }
    else
    {
      log_warning("can not find wintness function '{}'", inv.value);
      continue;
    }
  }

  if (options_.get_bool_option("witness-parse-tree"))
    // stop verify for debugging
    return true;

  return false;
}
