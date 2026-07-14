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
      if (
        entry["entry_type"] &&
        entry["entry_type"].as<std::string>() == "violation_sequence")
      {
        log_error(
          "Witness is a violation witness; use --validate-violation-witness "
          "instead.");
        abort();
      }
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
  if (wp.column != c_nonset)
    wp.column_id = irep_idt(integer2string(wp.column));
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

// Returns the 1-based column of the first '?' outside string/char literals and
// // line comments, or 0 if none exists.
size_t first_ternary_col_in_line(const std::string &text)
{
  enum class S
  {
    Code,
    Str,
    Chr
  } state = S::Code;
  for (size_t i = 0; i < text.size(); ++i)
  {
    char c = text[i];
    switch (state)
    {
    case S::Code:
      if (c == '/' && i + 1 < text.size() && text[i + 1] == '/')
        return 0;
      if (c == '"')
        state = S::Str;
      else if (c == '\'')
        state = S::Chr;
      else if (c == '?')
        return i + 1;
      break;
    case S::Str:
      if (c == '\\')
        ++i;
      else if (c == '"')
        state = S::Code;
      break;
    case S::Chr:
      if (c == '\\')
        ++i;
      else if (c == '\'')
        state = S::Code;
      break;
    }
  }
  return 0;
}
} // namespace

std::vector<waypoint> &yaml_parser::get_waypoints(const std::string &path)
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
      if (
        entry["entry_type"] &&
        entry["entry_type"].as<std::string>() == "invariant_set")
      {
        log_error(
          "Witness is a correctness witness; use "
          "--validate-correctness-witness instead.");
        abort();
      }
      const auto &content = entry["content"];
      if (!content || !content.IsSequence())
        continue;

      for (const auto &item : content)
      {
        const auto &seg = item["segment"];
        if (!seg || !seg.IsSequence())
          continue;

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
          }
          wp.segment_idx = seg_idx;
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

void yaml_parser::fill_columns(
  const std::string &src_path,
  std::vector<waypoint> &waypoints)
{
  std::vector<waypoint *> pending;
  for (auto &wp : waypoints)
  {
    if (
      wp.type == waypoint::branching && wp.action != waypoint::avoid &&
      wp.column == c_nonset && wp.line != c_nonset && wp.line > 0)
      pending.push_back(&wp);
  }
  if (pending.empty())
    return;

  std::ifstream in(src_path);
  if (!in)
    return;

  std::sort(
    pending.begin(), pending.end(), [](const waypoint *a, const waypoint *b) {
      return a->line < b->line;
    });

  size_t cur_line = 0;
  size_t pi = 0;
  std::string line_text;
  while (pi < pending.size() && std::getline(in, line_text))
  {
    ++cur_line;
    if (static_cast<size_t>(pending[pi]->line.to_int64()) != cur_line)
      continue;

    size_t col = first_ternary_col_in_line(line_text);
    while (pi < pending.size() &&
           static_cast<size_t>(pending[pi]->line.to_int64()) == cur_line)
    {
      if (col != 0)
      {
        waypoint &wp = *pending[pi];
        wp.column = BigInt(static_cast<long long>(col));
        wp.column_id = irep_idt(integer2string(wp.column));
      }
      ++pi;
    }
  }
}

waypoint yaml_parser::parse_waypoint(const YAML::Node &node)
{
  waypoint wp;

  if (node["type"])
    wp.type = waypoint_type_from_string(node["type"].as<std::string>());
  if (node["action"])
    wp.action = action_from_string(node["action"].as<std::string>());

  const auto &constraint = node["constraint"];
  if (constraint)
  {
    if (constraint["value"])
      wp.value = constraint["value"].as<std::string>();
    if (constraint["format"])
      wp.format = constraint["format"].as<std::string>();
  }

  const auto &loc = node["location"];
  if (loc)
  {
    if (loc["file_name"])
      wp.file = loc["file_name"].as<std::string>();
    if (loc["line"])
      wp.line = BigInt(loc["line"].as<std::string>().c_str(), 10);
    if (loc["column"])
      wp.column = BigInt(loc["column"].as<std::string>().c_str(), 10);
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
  // Inject assumption waypoints on the same physical line as the pointed
  // statement. This preserves physical line numbers for all subsequent lines,
  // so both SV-COMP mode (--sv-comp, physical line numbers) and regular runs
  // (logical line numbers via #line) see the correct line for each waypoint.
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

    auto it = by_line.find(line_num);
    if (it != by_line.end())
    {
      for (const waypoint *wp : it->second)
      {
        log_progress(
          "Injecting {} assumption at line {}: {}",
          wp->action == waypoint::avoid ? "avoid" : "follow",
          line_num,
          wp->value);
        out << "#line " << line_num << " \"" << original_path << "\"\n";
        out << "__ESBMC_witness_assume(" << wp->segment_idx << ", (_Bool)("
            << wp->value << "));\n";
        out << "#line " << line_num << " \"" << original_path << "\"\n";
      }
    }

    out << line_text << "\n";
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
