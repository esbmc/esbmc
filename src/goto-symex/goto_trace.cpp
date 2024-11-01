#include <cassert>
#include <cstring>
#include <goto-symex/goto_trace.h>
#include <goto-symex/printf_formatter.h>
#include <goto-symex/witnesses.h>

#include <regex>
#include <langapi/language_util.h>
#include <langapi/languages.h>
#include <util/arith_tools.h>
#include <util/std_types.h>
#include <ostream>
#include <iomanip>
#include <fstream>
#include <stack>
#include <set>
#include <map>
#include <nlohmann/json.hpp>


using json = nlohmann::json;

void goto_tracet::output(const class namespacet &ns, std::ostream &out) const
{
  for (const auto &step : steps)
    step.output(ns, out);
}

void goto_trace_stept::dump() const
{
  std::ostringstream oss;
  output(*migrate_namespace_lookup, oss);
  log_debug("goto-trace", "{}", oss.str());
}

void goto_trace_stept::output(const namespacet &ns, std::ostream &out) const
{
  switch (type)
  {
  case goto_trace_stept::ASSERT:
    out << "ASSERT";
    break;

  case goto_trace_stept::ASSUME:
    out << "ASSUME";
    break;

  case goto_trace_stept::ASSIGNMENT:
    out << "ASSIGNMENT";
    break;

  default:
    assert(false);
  }

  if (type == ASSERT || type == ASSUME)
    out << " (" << guard << ")";

  out << "\n";

  if (!pc->location.is_nil())
    out << pc->location << "\n";

  if (pc->is_goto())
    out << "GOTO   ";
  else if (pc->is_assume())
    out << "ASSUME ";
  else if (pc->is_assert())
    out << "ASSERT ";
  else if (pc->is_other())
    out << "OTHER  ";
  else if (pc->is_assign())
    out << "ASSIGN ";
  else if (pc->is_function_call())
    out << "CALL   ";
  else
    out << "(?)    ";

  out << "\n";

  if (pc->is_other() || pc->is_assign())
  {
    irep_idt identifier;

    if (!is_nil_expr(original_lhs))
      identifier = to_symbol2t(original_lhs).get_symbol_name();
    else
      identifier = to_symbol2t(lhs).get_symbol_name();

    out << "  " << identifier << " = " << from_expr(ns, identifier, value)
        << "\n";
  }
  else if (pc->is_assert())
  {
    if (!guard)
    {
      out << "Violated property:"
          << "\n";
      if (pc->location.is_nil())
        out << "  " << pc->location << "\n";

      if (!comment.empty())
        out << "  " << comment << "\n";
      out << "  " << from_expr(ns, "", pc->guard) << "\n";
      out << "\n";
    }
  }

  out << "\n";
}

void counterexample_value(
  std::ostream &out,
  const namespacet &ns,
  const expr2tc &lhs,
  const expr2tc &value)
{
  out << "  " << from_expr(ns, "", lhs);
  if (is_nil_expr(value))
    out << "(assignment removed)";
  else
  {
    out << " = " << from_expr(ns, "", value);

    // Don't print the bit-vector if we're running on integer/real mode
    if (is_constant_expr(value) && !config.options.get_bool_option("ir"))
    {
      std::string binary_value = "";
      if (is_bv_type(value))
      {
        binary_value = integer2binary(
          to_constant_int2t(value).value, value->type->get_width());
      }
      else if (is_fixedbv_type(value))
      {
        binary_value =
          to_constant_fixedbv2t(value).value.to_expr().get_value().as_string();
      }
      else if (is_floatbv_type(value))
      {
        binary_value =
          to_constant_floatbv2t(value).value.to_expr().get_value().as_string();
      }

      if (!binary_value.empty())
      {
        out << " (";

        std::string::size_type i = 0;
        for (const auto c : binary_value)
        {
          out << c;
          if (++i % 8 == 0 && binary_value.size() != i)
            out << ' ';
        }

        out << ")";
      }
    }

    out << "\n";
  }
}

void show_goto_trace_gui(
  std::ostream &out,
  const namespacet &ns,
  const goto_tracet &goto_trace)
{
  locationt previous_location;

  for (const auto &step : goto_trace.steps)
  {
    const locationt &location = step.pc->location;

    if ((step.type == goto_trace_stept::ASSERT) && !step.guard)
    {
      out << "FAILED"
          << "\n"
          << step.comment << "\n" // value
          << "\n"                 // PC
          << location.file() << "\n"
          << location.line() << "\n"
          << location.column() << "\n";
    }
    else if (step.type == goto_trace_stept::ASSIGNMENT)
    {
      irep_idt identifier;

      if (!is_nil_expr(step.original_lhs))
        identifier = to_symbol2t(step.original_lhs).get_symbol_name();
      else
        identifier = to_symbol2t(step.lhs).get_symbol_name();

      std::string value_string = from_expr(ns, identifier, step.value);

      const symbolt *symbol = ns.lookup(identifier);
      irep_idt base_name;
      if (symbol)
        base_name = symbol->name;

      out << "TRACE"
          << "\n";

      out << identifier << "," << base_name << ","
          << get_type_id(step.value->type) << "," << value_string << "\n"
          << step.step_nr << "\n"
          << step.pc->location.file() << "\n"
          << step.pc->location.line() << "\n"
          << step.pc->location.column() << "\n";
    }
    else if (location != previous_location)
    {
      // just the location

      if (!location.file().empty())
      {
        out << "TRACE"
            << "\n";

        out << "," // identifier
            << "," // base_name
            << "," // type
            << ""
            << "\n" // value
            << step.step_nr << "\n"
            << location.file() << "\n"
            << location.line() << "\n"
            << location.column() << "\n";
      }
    }

    previous_location = location;
  }
}

void show_state_header(
  std::ostream &out,
  const goto_trace_stept &state,
  const locationt &location,
  unsigned step_nr)
{
  out << "\n";
  out << "State " << step_nr;
  out << " " << location << " thread " << state.thread_nr << "\n";
  out << "----------------------------------------------------"
      << "\n";
}

void violation_graphml_goto_trace(
  optionst &options,
  const namespacet &ns,
  const goto_tracet &goto_trace)
{
  grapht graph(grapht::VIOLATION);
  graph.verified_file = options.get_option("input-file");

  log_progress("Generating Violation Witness for: {}", graph.verified_file);

  edget *first_edge = &graph.edges.at(0);
  nodet *prev_node = first_edge->to_node;

  for (const auto &step : goto_trace.steps)
  {
    switch (step.type)
    {
    case goto_trace_stept::ASSERT:
      if (!step.guard)
      {
        graph.check_create_new_thread(step.thread_nr, prev_node);
        prev_node = graph.edges.back().to_node;

        nodet *violation_node = new nodet();
        violation_node->violation = true;

        edget violation_edge(prev_node, violation_node);
        violation_edge.thread_id = std::to_string(step.thread_nr);
        violation_edge.start_line = get_line_number(
          graph.verified_file,
          std::atoi(step.pc->location.get_line().c_str()),
          options);

        graph.edges.push_back(violation_edge);

        /* having printed a property violation, don't print more steps. */

        graph.generate_graphml(options);
        return;
      }
      break;

    case goto_trace_stept::ASSIGNMENT:
      if (
        step.pc->is_assign() || step.pc->is_return() ||
        (step.pc->is_other() && is_nil_expr(step.lhs)) ||
        step.pc->is_function_call())
      {
        std::string assignment = get_formated_assignment(ns, step);

        graph.check_create_new_thread(step.thread_nr, prev_node);
        prev_node = graph.edges.back().to_node;

        edget new_edge;
        new_edge.thread_id = std::to_string(step.thread_nr);
        new_edge.assumption = assignment;
        new_edge.start_line = get_line_number(
          graph.verified_file,
          std::atoi(step.pc->location.get_line().c_str()),
          options);

        nodet *new_node = new nodet();
        new_edge.from_node = prev_node;
        new_edge.to_node = new_node;
        prev_node = new_node;
        graph.edges.push_back(new_edge);
      }
      break;

    default:
      continue;
    }
  }
}

void correctness_graphml_goto_trace(
  optionst &options,
  const namespacet &ns,
  const goto_tracet &goto_trace)
{
  grapht graph(grapht::CORRECTNESS);
  graph.verified_file = options.get_option("input-file");
  log_progress("Generating Correctness Witness for: {}", graph.verified_file);

  edget *first_edge = &graph.edges.at(0);
  nodet *prev_node = first_edge->to_node;

  for (const auto &step : goto_trace.steps)
  {
    /* checking restrictions for correctness GraphML */
    if (
      (!(is_valid_witness_step(ns, step))) ||
      (!(step.is_assume() || step.is_assert())))
      continue;

    std::string invariant = get_invariant(
      graph.verified_file,
      std::atoi(step.pc->location.get_line().c_str()),
      options);

    if (invariant.empty())
      continue; /* we don't have to consider this invariant */

    nodet *new_node = new nodet();
    edget *new_edge = new edget();
    std::string function = step.pc->location.get_function().c_str();
    new_edge->start_line = get_line_number(
      graph.verified_file,
      std::atoi(step.pc->location.get_line().c_str()),
      options);
    new_node->invariant = invariant;
    new_node->invariant_scope = function;

    new_edge->from_node = prev_node;
    new_edge->to_node = new_node;
    prev_node = new_node;
    graph.edges.push_back(*new_edge);
  }

  graph.generate_graphml(options);
}

void show_goto_trace(
  std::ostream &out,
  const namespacet &ns,
  const goto_tracet &goto_trace)
{
  using json = nlohmann::json;

  // Original trace output
  unsigned prev_step_nr = 0;
  bool first_step = true;

  // JSON data structure for coverage
  json test_data;
  test_data["type"] = "coverage_report";
  test_data["timestamp"] = std::time(nullptr);
  test_data["execution_path"] = json::array();
  test_data["files"] = json::object();

  // Load source files and initialize coverage data
  std::map<std::string, std::vector<std::string>> source_files;
  
  // First collect all files we need and initialize coverage
  for (const auto &step : goto_trace.steps) {
    if (!step.pc->location.is_nil()) {
      std::string current_file = step.pc->location.get_file().as_string();
      if (source_files.find(current_file) == source_files.end()) {
        std::ifstream source_file(current_file);
        if (source_file.is_open()) {
          std::vector<std::string> lines;
          std::string line;
          while (std::getline(source_file, line)) {
            lines.push_back(line);
          }
          source_files[current_file] = lines;
          
          // Initialize file data with all lines initially uncovered
          json coverage_data;
          for (size_t i = 0; i < lines.size(); i++) {
            coverage_data[std::to_string(i + 1)] = {
              {"covered", false},
              {"function", ""},
              {"hits", 0}
            };
          }
          
          test_data["files"][current_file] = {
            {"total_lines", lines.size()},
            {"source", lines},
            {"coverage", coverage_data},
            {"functions", json::object()}
          };
        }
      }
    }
  }

  // Process each step
  for (const auto &step : goto_trace.steps)
  {
    if (step.pc->location.is_nil())
      continue;

    json step_data;
    std::string current_file = step.pc->location.get_file().as_string();
    std::string current_line = step.pc->location.get_line().as_string();
    std::string current_function = step.pc->location.get_function().as_string();
    int line_num = std::stoi(current_line);

    step_data["file"] = current_file;
    step_data["line"] = line_num;
    step_data["function"] = current_function;

    // Mark line as covered and increment hit count
    if (test_data["files"].contains(current_file)) {
      auto& file_data = test_data["files"][current_file];
      auto& line_coverage = file_data["coverage"][current_line];
      line_coverage["covered"] = true;
      line_coverage["function"] = current_function;
      line_coverage["hits"] = line_coverage["hits"].get<int>() + 1;

      // Track function coverage
      if (!file_data["functions"].contains(current_function)) {
        file_data["functions"][current_function] = {
          {"first_line", current_line},
          {"covered_lines", json::array()},
          {"hits", 0}
        };
      }
      file_data["functions"][current_function]["hits"] = 
        file_data["functions"][current_function]["hits"].get<int>() + 1;
      if (std::find(file_data["functions"][current_function]["covered_lines"].begin(),
                    file_data["functions"][current_function]["covered_lines"].end(),
                    current_line) == file_data["functions"][current_function]["covered_lines"].end()) {
        file_data["functions"][current_function]["covered_lines"].push_back(current_line);
      }
    }

    switch (step.type)
    {
    case goto_trace_stept::ASSERT:
      step_data["type"] = "assertion";
      step_data["passed"] = step.guard;
      if (!step.guard) {
        step_data["comment"] = step.comment;
        if (step.pc->is_assert())
          step_data["assertion"] = from_expr(ns, "", step.pc->guard);
      }
      break;

    case goto_trace_stept::ASSIGNMENT:
      if (!is_nil_expr(step.lhs) && is_symbol2t(step.lhs)) {
        const symbol2t &sym = to_symbol2t(step.lhs);
        step_data["type"] = "assignment";
        step_data["variable"] = sym.thename.as_string();
        step_data["value"] = from_expr(ns, "", step.value);
      }
      break;

    case goto_trace_stept::OUTPUT:
      step_data["type"] = "output";
      {
        printf_formattert printf_formatter;
        printf_formatter(step.format_string, step.output_args);
        std::ostringstream output_stream;
        printf_formatter.print(output_stream);
        step_data["output"] = output_stream.str();
      }
      break;

    case goto_trace_stept::ASSUME:
      step_data["type"] = "assume";
      step_data["condition"] = from_expr(ns, "", step.pc->guard);
      break;

    default:
      break;
    }

    test_data["execution_path"].push_back(step_data);
  }

  // Calculate coverage statistics
  for (auto& [file, file_data] : test_data["files"].items()) {
    size_t covered_lines = 0;
    size_t total_lines = file_data["total_lines"].get<size_t>();
    
    // Count actually covered lines
    for (const auto& [line_num, coverage] : file_data["coverage"].items()) {
      if (coverage["covered"].get<bool>()) {
        covered_lines++;
      }
    }
    
    double coverage_percent = total_lines > 0 ? 
      (static_cast<double>(covered_lines) * 100.0) / static_cast<double>(total_lines) : 0.0;
    
    file_data["coverage_percentage"] = coverage_percent;
    file_data["covered_lines_count"] = covered_lines;
  }

  // Write JSON to file
  std::ofstream json_out("tests.json");
  json_out << std::setw(2) << test_data << std::endl;

  // Generate HTML report
  std::ofstream html_out("coverage_report.html");
  html_out << R"(
<!DOCTYPE html>
<html>
<head>
  <title>Code Coverage Report</title>
  <style>
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
      margin: 20px;
      background-color: #f8f9fa;
    }
    .covered { background-color: #90EE90; }
    .uncovered { background-color: #FFB6C6; }
    .line-number { 
      color: #666; 
      padding-right: 10px; 
      user-select: none;
      display: inline-block;
      width: 40px;
      text-align: right;
      border-right: 1px solid #ddd;
      margin-right: 10px;
    }
    pre { 
      margin: 0; 
      padding: 10px; 
      background: white;
      border-radius: 5px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .file-header { 
      background: white; 
      padding: 20px;
      margin: 20px 0;
      border-radius: 5px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .coverage-bar { 
      height: 20px;
      background: #FFB6C6;
      margin: 10px 0;
      border-radius: 10px;
      overflow: hidden;
    }
    .coverage-fill { 
      height: 100%;
      background: #90EE90;
      transition: width 0.3s ease;
    }
    .line { padding: 2px 5px; }
    .line:hover { background-color: rgba(0,0,0,0.05); }
    .function-coverage {
      margin-top: 15px;
      padding: 10px;
      background: #f8f9fa;
      border-radius: 5px;
    }
    .hits { 
      float: right;
      color: #666;
      font-size: 0.9em;
    }
  </style>
</head>
<body>
  <h1>Code Coverage Report</h1>
)";

  for (const auto& [file, file_data] : test_data["files"].items()) {
    html_out << "<div class='file-header'>\n";
    html_out << "<h2>File: " << file << "</h2>\n";
    double coverage_pct = file_data["coverage_percentage"].get<double>();
    size_t covered = file_data["covered_lines_count"].get<size_t>();
    size_t total = file_data["total_lines"].get<size_t>();
    
    html_out << "<p>Coverage: " << std::fixed << std::setprecision(2) 
             << coverage_pct << "% (" << covered << "/" << total << " lines)</p>\n";
    html_out << "<div class='coverage-bar'><div class='coverage-fill' style='width: "
             << coverage_pct << "%'></div></div>\n";

    // Add function coverage information
    html_out << "<div class='function-coverage'>\n";
    html_out << "<h3>Function Coverage:</h3>\n";
    for (const auto& [func_name, func_data] : file_data["functions"].items()) {
      html_out << "<p>" << func_name << ": " 
               << func_data["covered_lines"].size() << " lines covered "
               << "(hit " << func_data["hits"].get<int>() << " times)</p>\n";
    }
    html_out << "</div>\n";
    
    html_out << "</div>\n<pre>\n";

    const auto& coverage = file_data["coverage"];
    const auto& source = file_data["source"];
    
    for (size_t i = 0; i < source.size(); i++) {
      std::string line_num = std::to_string(i + 1);
      const auto& line_coverage = coverage[line_num];
      bool is_covered = line_coverage["covered"].get<bool>();
      int hits = line_coverage["hits"].get<int>();
      
      std::string line_class = is_covered ? "covered" : "uncovered";
      html_out << "<div class='line " << line_class << "'>"
               << "<span class='line-number'>" << line_num << "</span>"
               << source[i].get<std::string>();
      if (hits > 0) {
        html_out << "<span class='hits'>(" << hits << " hits)</span>";
      }
      html_out << "</div>\n";
    }
    html_out << "</pre>\n";
  }

  html_out << "</body></html>\n";

  // Output original format to out stream for backward compatibility
  for (const auto &step : goto_trace.steps)
    step.output(ns, out);
}

// End of file