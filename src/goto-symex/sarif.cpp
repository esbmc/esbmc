#include <goto-symex/sarif.h>

#include <ac_config.h>
#include <charconv>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <util/cwe_mapping.h>
#include <util/message.h>

using json = nlohmann::json;

namespace
{
unsigned parse_line(std::string_view s)
{
  // Non-throwing decimal parse. Returns 0 on empty / non-numeric input —
  // both are valid SARIF (region.startLine is optional, so we omit it in
  // the caller when the parse yields 0).
  unsigned v = 0;
  if (s.empty())
    return 0;
  auto res = std::from_chars(s.data(), s.data() + s.size(), v);
  return res.ec == std::errc{} ? v : 0u;
}
} // namespace

void sarif_goto_trace(
  const optionst &options,
  const namespacet & /*ns*/,
  const goto_tracet &goto_trace)
{
  const std::string out_path = options.get_option("sarif-output");
  if (out_path.empty())
    return;

  // Collect violation steps and the rules / CWE ids they exercise. The
  // substring-to-rule mapping comes from util/cwe_mapping — single source of
  // truth shared with the textual / JSON / GraphML outputs.
  struct result_t
  {
    std::string rule_id;
    std::string message;
    std::string file;
    unsigned line = 0;
    std::vector<unsigned> cwes;
  };
  std::vector<result_t> results;
  std::map<std::string, std::string> rule_descs; // id -> short description
  std::map<std::string, std::vector<unsigned>> rule_cwes; // id -> ids
  std::set<unsigned> all_cwes;

  for (const auto &step : goto_trace.steps)
  {
    if (step.type != goto_trace_stept::ASSERT || step.guard)
      continue;

    const cwe_rule_t &rule = cwe_rule_for(step.comment);
    result_t r;
    r.rule_id = rule.sarif_id;
    r.message = step.comment.empty() ? "Assertion check" : step.comment;
    r.file = step.pc->location.get_file().as_string();
    r.line = parse_line(step.pc->location.get_line().as_string());
    r.cwes = rule.cwes;

    rule_descs[r.rule_id] = rule.short_description;
    rule_cwes[r.rule_id] = r.cwes;
    for (unsigned id : r.cwes)
      all_cwes.insert(id);

    results.push_back(std::move(r));
  }

  // Build SARIF 2.1.0 document.
  json doc;
  doc["$schema"] =
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/cs01/schemas/"
    "sarif-schema-2.1.0.json";
  doc["version"] = "2.1.0";

  json run;
  run["tool"]["driver"]["name"] = "ESBMC";
  run["tool"]["driver"]["version"] = ESBMC_VERSION;
  run["tool"]["driver"]["informationUri"] = "https://esbmc.org";

  // SARIF §3.49.7: reportingDescriptor.name is a `simpleName` (no spaces,
  // letters/digits/period/underscore only). The human-readable text goes in
  // shortDescription; we omit `name` entirely because the stable identifier
  // is already `id`.
  json rules = json::array();
  for (const auto &[id, desc] : rule_descs)
  {
    json rule;
    rule["id"] = id;
    rule["shortDescription"]["text"] = desc;
    json tags = json::array();
    for (unsigned cwe : rule_cwes[id])
      tags.push_back("external/cwe/cwe-" + std::to_string(cwe));
    if (!tags.empty())
      rule["properties"]["tags"] = tags;
    rules.push_back(rule);
  }
  run["tool"]["driver"]["rules"] = rules;

  if (!all_cwes.empty())
  {
    json taxonomy;
    taxonomy["name"] = "CWE";
    taxonomy["organization"] = "MITRE";
    taxonomy["version"] = "4.20";
    taxonomy["informationUri"] = "https://cwe.mitre.org/";
    taxonomy["shortDescription"]["text"] = "Common Weakness Enumeration";
    json taxa = json::array();
    for (unsigned id : all_cwes)
    {
      json t;
      t["id"] = std::to_string(id);
      // taxon.name is also a simpleName; the CWE numeric id meets that, and
      // the MITRE title goes in shortDescription.
      std::string_view name = cwe_name(id);
      if (!name.empty())
        t["shortDescription"]["text"] = std::string(name);
      t["helpUri"] = "https://cwe.mitre.org/data/definitions/" +
                     std::to_string(id) + ".html";
      taxa.push_back(t);
    }
    taxonomy["taxa"] = taxa;
    run["taxonomies"] = json::array({taxonomy});
  }

  json results_json = json::array();
  for (const auto &r : results)
  {
    json result;
    result["ruleId"] = r.rule_id;
    result["level"] = "error";
    result["message"]["text"] = r.message;

    json loc;
    loc["physicalLocation"]["artifactLocation"]["uri"] = r.file;
    if (r.line > 0)
      loc["physicalLocation"]["region"]["startLine"] = r.line;
    result["locations"] = json::array({loc});

    if (!r.cwes.empty())
    {
      json taxa_refs = json::array();
      for (unsigned id : r.cwes)
      {
        json ref;
        ref["id"] = std::to_string(id);
        ref["toolComponent"]["name"] = "CWE";
        taxa_refs.push_back(ref);
      }
      result["taxa"] = taxa_refs;
    }

    results_json.push_back(result);
  }
  run["results"] = results_json;

  doc["runs"] = json::array({run});

  const std::string serialised = doc.dump(2);
  if (out_path == "-")
  {
    std::cout << serialised << "\n";
    return;
  }

  std::ofstream out(out_path);
  if (!out)
  {
    log_error("Could not open SARIF output file: {}", out_path);
    return;
  }
  out << serialised << "\n";
}
