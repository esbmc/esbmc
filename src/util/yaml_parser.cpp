#include <yaml_parser.h>
#include <fstream>
#include <unordered_map>
#include <sstream>
#include <util/message.h>

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
        invariant inv = parse_invariant(inv_node);
        if (inv.type != invariant::unknown)
          result.push_back(std::move(inv));
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

  log_warning("Unknown invariant type '{}', skipping invariant", s);
  return invariant::unknown;
}

namespace
{
void intern_location(waypoint &wp)
{
  if (wp.line != c_nonset)
    wp.line_id = irep_idt(integer2string(wp.line));
  wp.function_id = irep_idt(wp.function);
}

// Shared parse cache: get_waypoints and get_target_waypoint are both called
// on the same file per run, so parse once and reuse.
struct
{
  std::string path;
  std::vector<waypoint> waypoints;
  waypoint target;
  bool has_target = false;
} wp_cache;
} // namespace

std::vector<waypoint> yaml_parser::get_waypoints(const std::string &path)
{
  if (path == wp_cache.path)
    return wp_cache.waypoints;

  wp_cache.path = path;
  wp_cache.waypoints.clear();
  wp_cache.has_target = false;
  try
  {
    YAML::Node root = YAML::LoadFile(path);
    if (!root || !root.IsSequence())
      return wp_cache.waypoints;

    size_t seg_idx = 0;
    for (const auto &entry : root)
    {
      const auto &content = entry["content"];
      if (!content || !content.IsSequence())
        continue;

      for (const auto &item : content)
      {
        const auto &seg = item["segment"];
        if (!seg || !seg.IsSequence())
          continue;

        size_t wp_count = 0;
        for (const auto &wp_node : seg)
        {
          const auto &node = wp_node["waypoint"];
          if (!node)
            continue;
          waypoint wp = parse_waypoint(node);
          if (wp.type == waypoint::unknown)
            continue;
          intern_location(wp);
          if (wp.type == waypoint::target)
          {
            wp_cache.target = wp;
            wp_cache.has_target = true;
            continue;
          }
          wp.segment_idx = seg_idx;
          wp.wp_idx_in_seg = wp_count++;
          wp_cache.waypoints.push_back(std::move(wp));
        }
        ++seg_idx;
      }
    }
  }
  catch (const YAML::Exception &e)
  {
    log_error(
      "Failed to parse violation witness YAML '{}': {}", path, e.what());
    wp_cache.path.clear();
  }
  return wp_cache.waypoints;
}

bool yaml_parser::get_target_waypoint(const std::string &path, waypoint &out)
{
  get_waypoints(path);
  if (!wp_cache.has_target)
    return false;
  out = wp_cache.target;
  return true;
}

waypoint yaml_parser::parse_waypoint(const YAML::Node &node)
{
  waypoint wp;

  if (node["type"])
    wp.type = waypoint_type_from_string(node["type"].as<std::string>());
  if (node["action"])
    wp.action = action_from_string(node["action"].as<std::string>());

  const auto &constraint = node["constraint"];
  if (constraint && constraint["value"])
    wp.value = constraint["value"].as<std::string>();

  const auto &loc = node["location"];
  if (loc)
  {
    if (loc["file_name"])
      wp.file = loc["file_name"].as<std::string>();
    if (loc["line"])
      wp.line = BigInt(loc["line"].as<std::string>().c_str(), 10);
    if (loc["function"])
      wp.function = loc["function"].as<std::string>();
  }

  return wp;
}

waypoint::Type yaml_parser::waypoint_type_from_string(const std::string &s)
{
  if (s == "assumption")
    return waypoint::assumption;
  if (s == "target")
    return waypoint::target;
  if (s == "function_enter")
    return waypoint::function_enter;
  if (s == "function_return")
    return waypoint::function_return;
  if (s == "branching")
    return waypoint::branching;

  log_warning("Unknown waypoint type '{}', skipping waypoint", s);
  return waypoint::unknown;
}

waypoint::Action yaml_parser::action_from_string(const std::string &s)
{
  if (s == "follow")
    return waypoint::follow;
  if (s == "avoid")
    return waypoint::avoid;
  if (s == "cycle")
    return waypoint::cycle;

  log_warning("Unknown waypoint action '{}', treating as follow", s);
  return waypoint::follow;
}

std::string yaml_parser::build_violation_witness_source(
  const std::string &source_path,
  const std::string &original_path,
  const std::vector<waypoint> &waypoints)
{
  // Inject after the source line so declarations are in scope.
  std::unordered_map<size_t, std::vector<const waypoint *>> by_line;
  by_line.reserve(waypoints.size());
  for (const auto &wp : waypoints)
  {
    if (wp.type != waypoint::assumption || wp.line == c_nonset)
      continue;
    by_line[static_cast<size_t>(wp.line.to_int64())].push_back(&wp);
  }

  if (by_line.empty())
    return {};

  std::ifstream in(source_path);
  if (!in)
    return {};

  std::ostringstream out;
  std::string line_text;
  size_t line_num = 0;

  while (std::getline(in, line_text))
  {
    ++line_num;
    out << line_text << "\n";

    auto it = by_line.find(line_num);
    if (it != by_line.end())
    {
      out << "#line " << line_num << " \"" << original_path << "\"\n";
      for (const waypoint *wp : it->second)
      {
        out << "__ESBMC_witness_assume(" << wp->segment_idx << ", "
            << wp->wp_idx_in_seg << ", (_Bool)(" << wp->value << "));\n";
        log_progress(
          "Injecting {} assumption at line {}: {}",
          wp->action == waypoint::avoid ? "avoid" : "follow",
          line_num,
          wp->value);
      }
      out << "#line " << (line_num + 1) << " \"" << original_path << "\"\n";
    }
  }

  return out.str();
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
