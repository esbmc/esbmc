#include <goto-symex/sarif.h>

#include <ac_config.h>
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
struct rule_t
{
  const char *substring;
  const char *id;
  const char *name;
};

// Stable rule descriptors keyed off the violation comment. Order matches
// util/cwe_mapping.cpp: longest substring first.
const rule_t &rule_for(const std::string &comment)
{
  static const rule_t table[] = {
    {"dereference failure: invalidated dynamic object freed",
     "invalidated-dynamic-object-freed",
     "Freed dynamic object dereference"},
    {"dereference failure: invalid pointer freed",
     "invalid-pointer-freed",
     "Free of invalid pointer"},
    {"dereference failure: invalidated dynamic object",
     "invalidated-dynamic-object",
     "Dereference of invalidated dynamic object"},
    {"dereference failure: accessed expired variable pointer",
     "expired-pointer-dereference",
     "Expired pointer dereference"},
    {"dereference failure: free() of non-dynamic memory",
     "free-non-dynamic-memory",
     "free() of non-dynamic memory"},
    {"dereference failure: forgotten memory",
     "memory-leak",
     "Forgotten memory (leak)"},
    {"dereference failure: NULL pointer",
     "null-pointer-dereference",
     "NULL pointer dereference"},
    {"dereference failure: memset of memory segment",
     "memset-out-of-bounds",
     "memset out of bounds"},
    {"dereference failure on memcpy: reading memory segment",
     "memcpy-out-of-bounds",
     "memcpy out of bounds"},
    {"dereference failure: invalid pointer",
     "invalid-pointer-dereference",
     "Invalid pointer dereference"},
    {"Operand of free must have zero pointer offset",
     "free-non-zero-offset",
     "free() with non-zero pointer offset"},
    {"array bounds violated", "array-bounds-violated", "Array bounds violated"},
    {"Access to object out of bounds",
     "object-out-of-bounds",
     "Access to object out of bounds"},
    {"Same object violation",
     "same-object-violation",
     "Same-object pointer comparison violation"},
    {"Cast arithmetic overflow",
     "cast-arithmetic-overflow",
     "Cast arithmetic overflow"},
    {"arithmetic overflow", "arithmetic-overflow", "Arithmetic overflow"},
    {"division by zero", "division-by-zero", "Division by zero"},
    {"NaN on", "nan", "NaN result"},
    {"undefined behavior on shift operation",
     "shift-undefined-behavior",
     "Undefined behavior on shift"},
    {"atomicity violation", "atomicity-violation", "Atomicity violation"},
    {"data race on", "data-race", "Data race"},
    {"unreachable code reached",
     "reachable-error",
     "Reachable error/assertion"},
  };
  static const rule_t fallback{
    nullptr, "esbmc-assertion", "ESBMC assertion violation"};
  for (const auto &e : table)
    if (comment.find(e.substring) != std::string::npos)
      return e;
  return fallback;
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

  // Collect violation steps and the rules / CWE ids they exercise.
  struct result_t
  {
    std::string rule_id;
    std::string message;
    std::string file;
    unsigned line = 0;
    std::vector<unsigned> cwes;
  };
  std::vector<result_t> results;
  std::map<std::string, std::string> rule_names;          // id -> name
  std::map<std::string, std::vector<unsigned>> rule_cwes; // id -> ids
  std::set<unsigned> all_cwes;

  for (const auto &step : goto_trace.steps)
  {
    if (step.type != goto_trace_stept::ASSERT || step.guard)
      continue;

    const rule_t &rule = rule_for(step.comment);
    result_t r;
    r.rule_id = rule.id;
    r.message = step.comment.empty() ? "Assertion check" : step.comment;
    r.file = step.pc->location.get_file().as_string();
    if (!step.pc->location.get_line().empty())
      r.line = std::stoul(step.pc->location.get_line().as_string());
    r.cwes = cwe_for(step.comment);

    rule_names[r.rule_id] = rule.name;
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

  json rules = json::array();
  for (const auto &[id, name] : rule_names)
  {
    json rule;
    rule["id"] = id;
    rule["name"] = name;
    rule["shortDescription"]["text"] = name;
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
      std::string_view name = cwe_name(id);
      if (!name.empty())
      {
        t["name"] = std::string(name);
        t["shortDescription"]["text"] = std::string(name);
      }
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
