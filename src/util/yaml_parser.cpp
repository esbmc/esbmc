#include <yaml_parser.h>

yaml_parser::yaml_parser(const std::string &path) : file_path(path)
{
}

bool yaml_parser::load_file(const std::string &path)
{
  try
  {
    root = YAML::LoadFile(path);
  }
  catch (const YAML::Exception &e)
  {
    log_error("Failed to parse YAML file '{}': {}", path, e.what());
    return true;
  }

  return false;
}

std::vector<invariant> yaml_parser::get_invariants() const
{
  std::vector<invariant> result;
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
      invariant info = parse_invariant(inv_node);
      result.push_back(std::move(info));
    }
  }
  return result;
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
      info.line = loc["line"].as<BigInt>();
    if (loc["column"])
      info.column = loc["column"].as<BigInt>();

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

  return invariant::loop_invariant;
}
