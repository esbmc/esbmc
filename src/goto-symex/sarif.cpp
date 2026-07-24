#include <goto-symex/sarif.h>

#include <ac_config.h>
#include <charconv>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <util/base/cwe_mapping.h>
#include <util/message/message.h>

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

// The pieces below are the single source of truth for the SARIF document
// scaffolding shared by every emitter (sarif_goto_trace, sarif_dead_code): the
// driver block, the CWE taxonomy (pinned to 4.20) and the per-result taxa
// references. Keeping them here means a CWE-version bump or a schema change is
// a one-line edit rather than a hunt across parallel copies.

// A fresh `run` with the ESBMC tool.driver populated (rules added by caller).
json new_sarif_run()
{
  json run;
  run["tool"]["driver"]["name"] = "ESBMC";
  run["tool"]["driver"]["version"] = ESBMC_VERSION;
  run["tool"]["driver"]["informationUri"] = "https://esbmc.org";
  return run;
}

// The CWE taxonomy block (name/organization/version/taxa) for `cwes`, or a
// null json when the set is empty (caller should then omit `run.taxonomies`).
json cwe_taxonomy(const std::set<unsigned> &cwes)
{
  if (cwes.empty())
    return json();
  json taxonomy;
  taxonomy["name"] = "CWE";
  taxonomy["organization"] = "MITRE";
  taxonomy["version"] = "4.20";
  taxonomy["informationUri"] = "https://cwe.mitre.org/";
  taxonomy["shortDescription"]["text"] = "Common Weakness Enumeration";
  json taxa = json::array();
  for (unsigned id : cwes)
  {
    json t;
    t["id"] = std::to_string(id);
    // taxon.name is a SARIF simpleName; the CWE numeric id meets that, and the
    // MITRE title goes in shortDescription.
    std::string_view name = cwe_name(id);
    if (!name.empty())
      t["shortDescription"]["text"] = std::string(name);
    t["helpUri"] =
      "https://cwe.mitre.org/data/definitions/" + std::to_string(id) + ".html";
    taxa.push_back(t);
  }
  taxonomy["taxa"] = taxa;
  return taxonomy;
}

// The `result.taxa[]` references into the CWE taxonomy for `cwes`.
json cwe_taxa_refs(const std::vector<unsigned> &cwes)
{
  json taxa_refs = json::array();
  for (unsigned id : cwes)
  {
    json ref;
    ref["id"] = std::to_string(id);
    ref["toolComponent"]["name"] = "CWE";
    taxa_refs.push_back(ref);
  }
  return taxa_refs;
}

// A reportingDescriptor (a tool.driver.rules[] entry): stable id, the
// human-readable shortDescription, and the CWE ids exposed as
// `external/cwe/cwe-N` tags. SARIF §3.49.7 reserves reportingDescriptor.name
// for a `simpleName` (letters/digits/period/underscore); the human-readable
// text goes in shortDescription and we omit `name` since `id` is already the
// stable identifier.
json sarif_rule(
  const std::string &id,
  const std::string &short_description,
  const std::vector<unsigned> &cwes)
{
  json rule;
  rule["id"] = id;
  rule["shortDescription"]["text"] = short_description;
  json tags = json::array();
  for (unsigned cwe : cwes)
    tags.push_back("external/cwe/cwe-" + std::to_string(cwe));
  if (!tags.empty())
    rule["properties"]["tags"] = tags;
  return rule;
}

// A single SARIF result: rule id, level ("error" / "note"), message, physical
// location, and the per-result CWE taxa references.
json sarif_result(
  const std::string &rule_id,
  const std::string &level,
  const std::string &message,
  const std::string &file,
  unsigned line,
  const std::vector<unsigned> &cwes)
{
  json result;
  result["ruleId"] = rule_id;
  result["level"] = level;
  result["message"]["text"] = message;

  json loc;
  loc["physicalLocation"]["artifactLocation"]["uri"] = file;
  if (line > 0)
    loc["physicalLocation"]["region"]["startLine"] = line;
  result["locations"] = json::array({loc});

  if (!cwes.empty())
    result["taxa"] = cwe_taxa_refs(cwes);

  return result;
}

// Wrap `run` in a SARIF 2.1.0 document and write it to `out_path` ("-" is
// stdout). Shared serialisation so the schema URI lives in one place.
void write_sarif_document(const std::string &out_path, json run)
{
  json doc;
  doc["$schema"] =
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/cs01/schemas/"
    "sarif-schema-2.1.0.json";
  doc["version"] = "2.1.0";
  doc["runs"] = json::array({std::move(run)});

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
} // namespace

void sarif_goto_trace(
  const optionst &options,
  const namespacet & /*ns*/,
  const goto_tracet &goto_trace,
  const std::vector<dead_store_advisoryt> &advisories)
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
    std::string level = "error";
    std::string message;
    std::string file;
    unsigned line = 0;
    std::vector<unsigned> cwes;
  };
  std::vector<result_t> results;
  std::map<std::string, std::string> rule_descs; // id -> short description
  std::map<std::string, std::vector<unsigned>> rule_cwes; // id -> ids
  std::set<unsigned> all_cwes;

  auto record_rule = [&](const result_t &r, const cwe_rule_t &rule) {
    rule_descs[r.rule_id] = rule.short_description;
    rule_cwes[r.rule_id] = r.cwes;
    for (unsigned id : r.cwes)
      all_cwes.insert(id);
  };

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

    record_rule(r, rule);
    results.push_back(std::move(r));
  }

  // Dead-store advisories (CWE-563) are emitted as note-level results and do
  // not affect the verdict.
  for (const auto &adv : advisories)
  {
    const cwe_rule_t &rule = cwe_rule_for(adv.comment);
    result_t r;
    r.rule_id = rule.sarif_id;
    r.level = "note";
    r.message = adv.comment;
    r.file = adv.file;
    r.line = adv.line;
    r.cwes = rule.cwes;

    record_rule(r, rule);
    results.push_back(std::move(r));
  }

  // Build SARIF 2.1.0 document from shared scaffolding.
  json run = new_sarif_run();

  json rules = json::array();
  for (const auto &[id, desc] : rule_descs)
    rules.push_back(sarif_rule(id, desc, rule_cwes[id]));
  run["tool"]["driver"]["rules"] = rules;

  if (json taxonomy = cwe_taxonomy(all_cwes); !taxonomy.is_null())
    run["taxonomies"] = json::array({std::move(taxonomy)});

  json results_json = json::array();
  for (const auto &r : results)
    results_json.push_back(
      sarif_result(r.rule_id, r.level, r.message, r.file, r.line, r.cwes));
  run["results"] = results_json;

  write_sarif_document(out_path, std::move(run));
}

void sarif_dead_code(
  const optionst &options,
  const std::vector<dead_code_finding_t> &findings)
{
  const std::string out_path = options.get_option("sarif-output");
  if (out_path.empty())
    return;
  // A clean run (no findings) still emits a valid SARIF document with an empty
  // `results` array: consumers expect a well-formed run, not a missing file.

  // Single source of truth: the dead-code rule (id / description / CWE-561)
  // comes from util/cwe_mapping, shared with the textual output. It is a
  // dedicated accessor, not a cwe_rule_for() substring lookup, so CWE-561 never
  // leaks into ordinary violation mapping (issue #4495).
  const cwe_rule_t &rule = dead_code_cwe_rule();

  json run = new_sarif_run();

  run["tool"]["driver"]["rules"] =
    json::array({sarif_rule(rule.sarif_id, rule.short_description, rule.cwes)});

  const std::set<unsigned> cwes(rule.cwes.begin(), rule.cwes.end());
  if (json taxonomy = cwe_taxonomy(cwes); !taxonomy.is_null())
    run["taxonomies"] = json::array({std::move(taxonomy)});

  // Advisory findings are emitted at "note" level: the dead-code verdict never
  // flips a run to FAILED (issue #4495).
  json results_json = json::array();
  for (const auto &f : findings)
    results_json.push_back(sarif_result(
      rule.sarif_id,
      "note",
      f.message.empty() ? "Dead code" : f.message,
      f.file,
      f.line,
      rule.cwes));
  run["results"] = results_json;

  write_sarif_document(out_path, std::move(run));
}
