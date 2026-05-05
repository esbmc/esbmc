#include <yaml_parser.h>
#include <fstream>
#include <unordered_map>
#include <sstream>

std::vector<invariant> yaml_parser::read_invariants(const std::string &path)
{
  std::vector<invariant> result;
  try
  {
    YAML::Node root = YAML::LoadFile(path);
    if (!root || !root.IsSequence())
      return result;

    for (const auto &entry : root)
    {
      const auto &content = entry["content"];
      if (!content || !content.IsSequence())
        continue;

      for (const auto &c : content)
      {
        const auto &inv_node = c["invariant"];
        if (!inv_node)
          continue;
        result.push_back(parse_invariant(inv_node));
      }
    }
  }
  catch (const YAML::Exception &e)
  {
    log_error("Failed to parse witness YAML '{}': {}", path, e.what());
  }
  return result;
}

invariant yaml_parser::parse_invariant(const YAML::Node &node)
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

invariant::Type yaml_parser::type_from_string(const std::string &s)
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

std::string yaml_parser::build_injected_source(
  const std::string &source_path,
  const std::string &original_path,
  const std::vector<invariant> &invariants)
{
  std::unordered_map<size_t, std::vector<const invariant *>> by_line;
  by_line.reserve(invariants.size());
  for (const auto &inv : invariants)
    by_line[static_cast<size_t>(inv.line.to_int64())].push_back(&inv);

  std::ifstream in(source_path);
  if (!in)
    return {};

  std::ostringstream out;
  std::string line_text;
  size_t line_num = 0;

  while (std::getline(in, line_text))
  {
    ++line_num;
    auto it = by_line.find(line_num);
    if (it != by_line.end())
    {
      for (const invariant *inv : it->second)
      {
        switch (inv->type)
        {
        case invariant::loop_invariant:
          out << "__ESBMC_loop_invariant((_Bool)(" << inv->value << "));\n";
          log_progress(
            "Injecting loop invariant at line {}: {}", line_num, inv->value);
          break;
        case invariant::location_invariant:
          out << "__ESBMC_assert((_Bool)(" << inv->value
              << "), \"witness invariant\");\n";
          out << "__ESBMC_assume((_Bool)(" << inv->value << "));\n";
          log_progress(
            "Injecting location invariant at line {}: {}",
            line_num,
            inv->value);
          break;
        default:
          log_warning(
            "Unsupported invariant type for '{}', skipping", inv->value);
          break;
        }
      }
      out << "#line " << line_num << " \"" << original_path << "\"\n";
    }
    out << line_text << "\n";
  }

  return out.str();
}
