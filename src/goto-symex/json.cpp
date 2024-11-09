#include <util/message.h>
#include <goto-symex/goto_trace.h>
#include <util/language.h>
#include <langapi/language_util.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <regex>

using json = nlohmann::json;

namespace {
    // Helper function to check if string starts with prefix, as CI system prefers starts_with
    bool starts_with(const std::string& str, const std::string& prefix) {
        return str.size() >= prefix.size() && 
               str.compare(0, prefix.size(), prefix) == 0;
    }

    // Helper function to read file lines
    std::vector<std::string> read_file_lines(const std::string& filename) {
        std::vector<std::string> lines;
        std::ifstream input(filename);
        std::string line;
        
        while(std::getline(input, line)) {
            lines.push_back(std::move(line));
        }
        
        return lines;
    }

    std::string type_id_to_string(type2t::type_ids id) {
        switch(id) {
            case type2t::bool_id: return "bool";
            case type2t::empty_id: return "empty";
            case type2t::symbol_id: return "symbol";
            case type2t::struct_id: return "struct";
            case type2t::union_id: return "union";
            case type2t::pointer_id: return "pointer";
            case type2t::array_id: return "array";
            case type2t::code_id: return "code";
            case type2t::fixedbv_id: return "fixedbv";
            case type2t::unsignedbv_id: return "unsigned";
            case type2t::signedbv_id: return "signed";
            default: return fmt::format("unknown({})", static_cast<int>(id));
        }
    }

    std::string get_type_name(const type2tc& type) {
        if(is_struct_type(type)) {
            const struct_type2t& struct_type = to_struct_type(type);
            return "struct:" + id2string(struct_type.name);
        }
        else if(is_pointer_type(type)) {
            return "pointer";
        }
        else if(is_array_type(type)) {
            return "array";
        }
        else if(is_bool_type(type)) {
            return "bool";
        }
        else if(is_code_type(type)) {
            return "function";
        }
        else if(is_union_type(type)) {
            return "union";
        }
        else if(is_fixedbv_type(type)) {
            return "float";
        }
        else if(is_unsignedbv_type(type)) {
            return "unsigned";
        }
        else if(is_signedbv_type(type)) {
            return "signed";
        }
        else {
            return "unknown";
        }
    }

    std::set<std::string> find_included_headers(const std::string& file_path, std::set<std::string>& processed_files) {
        if (processed_files.find(file_path) != processed_files.end() || 
            starts_with(std::filesystem::path(file_path).string(), "/usr/")) {
            return {};
        }
        processed_files.insert(file_path);
        
        std::set<std::string> headers;
        std::ifstream source_file(file_path);
        
        if (!source_file.is_open()) {
            return headers;
        }

        std::regex include_pattern(R"(#\s*include\s*([<"])([\w./]+)[>"])");
        std::string line;
        
        while (std::getline(source_file, line)) {
            std::smatch match;
            if (std::regex_search(line, match, include_pattern)) {
                std::string header = match[2].str();
                
                // Try relative to current file
                std::filesystem::path local_path = std::filesystem::path(file_path).parent_path() / header;
                if (std::filesystem::exists(local_path)) {
                    auto resolved_path = std::filesystem::canonical(local_path).string();
                    if (!starts_with(resolved_path, "/usr/")) {
                        headers.insert(resolved_path);
                        auto nested_headers = find_included_headers(resolved_path, processed_files);
                        headers.insert(nested_headers.begin(), nested_headers.end());
                    }
                }
            }
        }
        
        return headers;
    }

    json serialize_value(const namespacet& ns, const expr2tc& expr, std::set<std::string>& seen) {
        if(is_nil_expr(expr)) {
            return nullptr;
        }

        try {
            if(is_pointer_type(expr->type)) {
                const pointer_type2t& ptr_type = to_pointer_type(expr->type);
                std::string addr_str = from_expr(ns, "", expr);
                
                if(!seen.insert(addr_str).second) {
                    return {
                        {"__type", "pointer"},
                        {"address", addr_str},
                        {"circular", true}
                    };
                }
                
                try {
                    expr2tc deref_expr = dereference2tc(ptr_type.subtype, expr);
                    if(!is_nil_expr(deref_expr)) {
                        return {
                            {"__type", "pointer"},
                            {"address", addr_str},
                            {"value", serialize_value(ns, deref_expr, seen)}
                        };
                    }
                } catch(const std::runtime_error&) {
                    // Ignore dereference errors
                }
                
                return {
                    {"__type", "pointer"},
                    {"address", addr_str},
                    {"value", nullptr}
                };
            }
            else if(is_struct_type(expr->type)) {
                const struct_type2t& struct_type = to_struct_type(expr->type);
                
                json struct_data = json::object();
                struct_data["__type"] = "struct";
                struct_data["name"] = id2string(struct_type.name);
                struct_data["members"] = json::object();
                
                for(size_t i = 0; i < struct_type.members.size(); i++) {
                    const irep_idt& member_name = struct_type.member_names[i];
                    
                    try {
                        expr2tc member_expr = member2tc(
                            struct_type.members[i],
                            expr,
                            member_name
                        );
                        struct_data["members"][id2string(member_name)] = 
                            serialize_value(ns, member_expr, seen);
                    } catch(const std::runtime_error&) {
                        struct_data["members"][id2string(member_name)] = nullptr;
                    }
                }
                return struct_data;
            }
            else {
                return from_expr(ns, "", expr);
            }
        }
        catch(const std::runtime_error& e) {
            log_status("Serialization error: {}", e.what());
            return nullptr;
        }
    }

    json get_assignment_json(const namespacet& ns, const expr2tc& lhs, const expr2tc& value) {
        std::set<std::string> seen;
        json assignment;
        
        assignment["lhs"] = from_expr(ns, "", lhs);
        assignment["lhs_type"] = type_id_to_string(lhs->type->type_id);
        
        if(!is_nil_expr(value)) {
            assignment["rhs"] = serialize_value(ns, value, seen);
            assignment["rhs_type"] = type_id_to_string(value->type->type_id);
        } else {
            assignment["rhs"] = nullptr;
            assignment["rhs_type"] = "nil";
        }
        
        return assignment;
    }

    void add_coverage_to_json(const goto_tracet& goto_trace, const namespacet& ns) {
        json test_entry;
        test_entry["steps"] = json::array();
        test_entry["status"] = "unknown";
        test_entry["coverage"] = {{"files", json::object()}};
        
        json initial_values = json::object();
        bool initial_state_captured = false;
        
        std::map<std::string, std::map<int, int>> line_hits;
        std::set<std::pair<std::string, int>> violations;
        std::set<std::string> all_referenced_files;
        std::set<std::string> processed_files;
        std::set<std::string> processed_vars;
        
        size_t step_count = 0;
        bool found_violation = false;

        // Collect referenced files
        for(const auto& step : goto_trace.steps) {
            if(step.pc != goto_programt::const_targett() && !step.pc->location.is_nil()) {
                const locationt& loc = step.pc->location;
                std::string file = id2string(loc.get_file());
                
                if(!file.empty() && !starts_with(file, "/usr/")) {
                    all_referenced_files.insert(file);
                    auto included_headers = find_included_headers(file, processed_files);
                    all_referenced_files.insert(included_headers.begin(), included_headers.end());
                }
            }
        }

        // Process steps
        for(const auto& step : goto_trace.steps) {
            if(step.pc != goto_programt::const_targett() && !step.pc->location.is_nil()) {
                const locationt& loc = step.pc->location;
                std::string file = id2string(loc.get_file());
                
                if(starts_with(file, "/usr/")) 
                    continue;
                
                std::string line_str = id2string(loc.get_line());
                std::string function = id2string(loc.get_function());
                
                try {
                    int line = std::stoi(line_str);
                    
                    if(line > 0) {
                        line_hits[file][line]++;

                        json step_data;
                        step_data["file"] = file;
                        step_data["line"] = line_str;
                        step_data["function"] = function;
                        step_data["step_number"] = step_count++;

                        if(step.is_assert()) {
                            if(!step.guard) {
                                violations.insert({file, line});
                                found_violation = true;
                                step_data["type"] = "violation";
                                step_data["message"] = step.comment.empty() ? "Assertion check" : step.comment;
                                test_entry["status"] = "violation";
                                step_data["assertion"] = {
                                    {"violated", true},
                                    {"comment", step.comment},
                                    {"guard", from_expr(ns, "", step.pc->guard)}
                                };
                            } else {
                                step_data["type"] = "assert";
                            }
                        }
                        else if(step.is_assume()) {
                            step_data["type"] = "assume";
                            step_data["message"] = "Assumption restriction";
                        }
                        else if(step.is_assignment()) {
                            step_data["type"] = "assignment";
                            std::string var_name = from_expr(ns, "", step.lhs);
                            
                            step_data["assignment"] = get_assignment_json(ns, step.lhs, step.value);

                            if(!initial_state_captured && processed_vars.find(var_name) == processed_vars.end()) {
                                processed_vars.insert(var_name);
                                std::set<std::string> seen_values;
                                json value_info = {
                                    {"name", var_name},
                                    {"type", type_id_to_string(step.lhs->type->type_id)},
                                    {"value", serialize_value(ns, step.value, seen_values)}
                                };
                                initial_values[var_name] = value_info;
                            }
                        }
                        else if(step.pc->is_function_call()) {
                            step_data["type"] = "function_call";
                            std::set<std::string> seen_args;
                            step_data["function_call"] = {
                                {"argument", from_expr(ns, "", step.lhs)},
                                {"value", serialize_value(ns, step.value, seen_args)}
                            };
                        }
                        else {
                            step_data["type"] = "other";
                        }

                        test_entry["steps"].push_back(step_data);
                    }
                } catch(std::exception& e) {
                    log_status("Error processing step: {}", e.what());
                    continue;
                }
            }
            
            if(step_count > 3) {
                initial_state_captured = true;
            }
        }

        test_entry["initial_values"] = initial_values;

        if(!found_violation && test_entry["status"] == "unknown") {
            test_entry["status"] = "success";
        }

        // Build coverage data
        for(const auto& file : all_referenced_files) {
            if(starts_with(file, "/usr/")) 
                continue;
            
            json file_coverage;
            file_coverage["covered_lines"] = json::object();

            if(line_hits.find(file) != line_hits.end()) {
                for(const auto& [line, hits] : line_hits[file]) {
                    std::string line_str = std::to_string(line);
                    bool is_violation = violations.find({file, line}) != violations.end();

                    file_coverage["covered_lines"][line_str] = {
                        {"covered", true},
                        {"hits", hits},
                        {"type", is_violation ? "violation" : "execution"}
                    };
                }

                file_coverage["coverage_stats"] = {
                    {"covered_lines", line_hits[file].size()},
                    {"total_hits", std::accumulate(line_hits[file].begin(), line_hits[file].end(), 0,
                        [](int sum, const auto& p) { return sum + p.second; })}
                };
            } else {
                file_coverage["coverage_stats"] = {
                    {"covered_lines", 0},
                    {"total_hits", 0}
                };
            }

            test_entry["coverage"]["files"][file] = file_coverage;
        }

        // Read existing JSON and append new test
        json all_tests = []() {
            std::ifstream input("report.json");
            if(input.is_open()) {
                try {
                    json existing;
                    input >> existing;
                    return existing;
                } catch(json::parse_error& e) {
                    log_status("Error parsing existing report.json: {}", e.what());
                }
            }
            return json::array();
        }();

        // Handle source files for first entry
        if(all_tests.empty()) {
            json source_data;
            for(const auto& file : all_referenced_files) {
                if(!starts_with(file, "/usr/")) {
                    try {
                        source_data[file] = read_file_lines(file);
                    } catch(std::exception& e) {
                        log_status("Error reading file {}: {}", file, e.what());
                        source_data[file] = std::vector<std::string>();
                        }
                }
            }
            test_entry["source_files"] = source_data;
        }

        all_tests.push_back(test_entry);

        // Write updated JSON
        std::ofstream json_out("report.json");
        if(!json_out.is_open()) {
            log_status("Error: Could not open report.json for writing");
            return;
        }
        json_out << std::setw(2) << all_tests << std::endl;
    }
} // anonymous namespace

void generate_json_report(
    const std::string_view uuid,
    const namespacet& ns, 
    const goto_tracet& goto_trace,
    const cmdlinet::options_mapt& options_map)
{
    log_status("Generating JSON report for trace: {}", uuid);
    add_coverage_to_json(goto_trace, ns);
}