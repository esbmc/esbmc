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
  out << "Starting trace output with " << goto_trace.steps.size() << " steps...\n";
  
  try {
    using json = nlohmann::json;
    json test_data;
    
    // Basic initialization
    test_data["type"] = "coverage_report";
    test_data["timestamp"] = std::time(nullptr);
    test_data["execution_path"] = json::array();
    test_data["files"] = json::object();
    
    std::map<std::string, std::vector<std::string>> source_files;
    
    out << "Starting step processing...\n";
    
    // Process steps
    size_t step_count = 0;
    for (const auto &step : goto_trace.steps)
    {
      step_count++;
      out << "\nProcessing step " << step_count << " of type " << static_cast<int>(step.type) << "\n";
      
      try {
        if (step.pc->location.is_nil()) {
          out << "Step has nil location, skipping...\n";
          continue;
        }
        
        std::string current_file = step.pc->location.get_file().as_string();
        out << "File: " << current_file << "\n";
        
        if (current_file.empty()) {
          out << "Empty file name, skipping...\n";
          continue;
        }
        
        // Load source file if not already loaded
        if (source_files.find(current_file) == source_files.end()) {
          out << "Loading new source file: " << current_file << "\n";
          std::ifstream source_file(current_file);
          if (source_file.is_open()) {
            std::vector<std::string> lines;
            std::string line;
            while (std::getline(source_file, line)) {
              lines.push_back(line);
            }
            source_files[current_file] = lines;
            out << "Loaded " << lines.size() << " lines from " << current_file << "\n";
            
            // Initialize file data in JSON
            test_data["files"][current_file] = {
              {"total_lines", lines.size()},
              {"source", lines},
              {"coverage", json::object()},
              {"functions", json::object()}
            };
          } else {
            out << "Failed to open source file: " << current_file << "\n";
          }
        }
        
        std::string current_line = step.pc->location.get_line().as_string();
        out << "Line: " << current_line << "\n";
        
        if (!current_line.empty()) {
          std::string current_function = step.pc->location.get_function().as_string();
          out << "Function: " << current_function << "\n";
          
          try {
            int line_num = std::stoi(current_line);
            auto& file_data = test_data["files"][current_file];
            
            // Update coverage data
            file_data["coverage"][current_line] = {
              {"covered", true},
              {"function", current_function},
              {"hits", file_data["coverage"].contains(current_line) ? 
                file_data["coverage"][current_line]["hits"].get<int>() + 1 : 1}
            };
            
            // Update function data
            if (!current_function.empty()) {
              if (!file_data["functions"].contains(current_function)) {
                file_data["functions"][current_function] = {
                  {"covered_lines", json::array()},
                  {"hits", 0}
                };
              }
              auto& func_data = file_data["functions"][current_function];
              func_data["hits"] = func_data["hits"].get<int>() + 1;
              
              // Add line to covered lines if not already present
              auto& covered_lines = func_data["covered_lines"];
              if (std::find(covered_lines.begin(), covered_lines.end(), current_line) == covered_lines.end()) {
                covered_lines.push_back(current_line);
              }
            }
          } catch (const std::exception& e) {
            out << "Error processing line number: " << e.what() << "\n";
          }
        }
        
      } catch (const std::exception& e) {
        out << "Exception processing step: " << e.what() << "\n";
      } catch (...) {
        out << "Unknown exception processing step\n";
      }
    }
    
    out << "Calculating coverage statistics...\n";
    
    // Calculate coverage statistics
    for (auto& [file, file_data] : test_data["files"].items()) {
      try {
        size_t total_lines = file_data["total_lines"].get<size_t>();
        size_t covered_lines = 0;
        
        for (const auto& [line, coverage] : file_data["coverage"].items()) {
          if (coverage["covered"].get<bool>()) {
            covered_lines++;
          }
        }
        
        double coverage_percent = total_lines > 0 ? 
          (static_cast<double>(covered_lines) * 100.0) / static_cast<double>(total_lines) : 0.0;
        
        file_data["coverage_percentage"] = coverage_percent;
        file_data["covered_lines_count"] = covered_lines;
        
        out << "File " << file << " coverage: " << coverage_percent << "% ("
            << covered_lines << "/" << total_lines << " lines)\n";
            
      } catch (const std::exception& e) {
        out << "Error calculating coverage for file " << file << ": " << e.what() << "\n";
      }
    }
    
    out << "Writing JSON output...\n";
    
    // Write JSON output
    std::ofstream json_out("tests.json");
    if (json_out.is_open()) {
      json_out << std::setw(2) << test_data << std::endl;
      out << "JSON file written successfully\n";
    } else {
      out << "Failed to open JSON output file\n";
    }
    
    out << "Writing original trace format...\n";
    
    // Output original format
    for (const auto &step : goto_trace.steps) {
      try {
        step.output(ns, out);
      } catch (const std::exception& e) {
        out << "Exception in original trace output: " << e.what() << "\n";
      } catch (...) {
        out << "Unknown exception in original trace output\n";
      }
    }
    
    out << "Trace output completed successfully\n";
    
  } catch (const std::exception& e) {
    out << "Top-level exception: " << e.what() << "\n";
    
    // Attempt original output on error
    for (const auto &step : goto_trace.steps) {
      try {
        step.output(ns, out);
      } catch (...) {
        out << "Failed to output step in error handler\n";
      }
    }
  } catch (...) {
    out << "Unknown top-level exception\n";
  }
}

// End of file