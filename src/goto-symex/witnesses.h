/*****************************************
 * GraphML Generation
 *****************************************
 *
 *  Modifications:
 *
 *  23/01/2016 - Updated for svcomp16 according to
 *  http://sv-comp.sosy-lab.org/2016/witnesses/s3_cln1_false.witness.cpachecker.graphml
 *
 */

#include <boost/property_tree/detail/ptree_implementation.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ptree_fwd.hpp>
#include <boost/property_tree/string_path.hpp>
#include <stdio.h>
#include <fstream>
#include <iterator>
#include <map>
#include <string>

typedef struct graph_props
{
  std::string sourcecodeLanguage;
  std::string witnessType;
} graph_p;

typedef struct node_props
{
  std::string nodeType = "";
  bool isFrontierNode = false;
  bool isViolationNode = false;
  bool isEntryNode = false;
  bool isSinkNode = false;
  std::string invariant;
  std::string invariantScope;
} node_p;

typedef struct edge_props
{
  std::string assumption = "";
  std::string sourcecode = "";
  int startline = -1;
  int endline = -1;
  std::string originFileName = "";
  std::string enterFunction = "";
  std::string returnFromFunction = "";
} edge_p;

int node_count;
int edge_count;

std::string execute_cmd(std::string command)
{
  /* add ./ for linux execution */
  std::string initial = command.substr(0, 1);
  if (initial != "./")
  {
    command = "./" + command;
  }
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe)
    return "ERROR";
  char buffer[128];
  std::string result = "";
  while (!feof(pipe))
  {
    if (fgets(buffer, 128, pipe) != NULL)
      result += buffer;
  }
  pclose(pipe);
  return result;
}

std::string read_file(std::string path)
{
  std::ifstream t(path.c_str());
  std::string str((std::istreambuf_iterator<char>(t)),
      std::istreambuf_iterator<char>());
  return str;
}

void write_file(std::string path, std::string content)
{
  std::ofstream out(path.c_str());
  out << content;
  out.close();
}

void generate_tokens(std::string tokenized_line,
    std::map<int, std::string> & tokens, int & token_index)
{
  std::istringstream tl_stream(tokenized_line.c_str());
  std::string line;
  while (std::getline(tl_stream, line))
  {
    if (line != "\n" && line != "")
    {
      tokens[token_index] = line;
      token_index++;
    }
  }
}

std::string trim(std::string str)
{
  const std::string whitespace_characters = " \t\r\n";
  size_t first_non_whitespace = str.find_first_not_of(whitespace_characters);
  if (first_non_whitespace == std::string::npos)
    return "";
  size_t last_non_whitespace = str.find_last_not_of(whitespace_characters);
  size_t length = last_non_whitespace - first_non_whitespace + 1;
  return str.substr(first_non_whitespace, length);
}

void map_line_number_to_content(std::string source_code_file,
    std::map<int, std::string> & line_content_map)
{
  std::ifstream sfile(source_code_file);
  if (!sfile)
  {
    return;
  }
  std::string source_content = read_file(source_code_file);
  std::istringstream source_stream(source_content.c_str());
  std::string line;
  int line_count = 0;
  while (std::getline(source_stream, line))
  {
    line_count++;
    line_content_map[line_count] = trim(line);
  }
}

void create_node(boost::property_tree::ptree & node, node_p & node_props)
{
  node.add("<xmlattr>.id", "n" + std::to_string(node_count++));
  if (!node_props.nodeType.empty())
  {
    boost::property_tree::ptree data_nodetype;
    data_nodetype.add("<xmlattr>.key", "notetype");
    data_nodetype.put_value(node_props.nodeType);
    node.add_child("data", data_nodetype);
  }
  if (node_props.isViolationNode != 0)
  {
    boost::property_tree::ptree data_violation;
    data_violation.add("<xmlattr>.key", "violation");
    data_violation.put_value("true");
    node.add_child("data", data_violation);
  }
  if (node_props.isSinkNode != 0)
  {
    boost::property_tree::ptree data_sink;
    data_sink.add("<xmlattr>.key", "sink");
    data_sink.put_value("true");
    node.add_child("data", data_sink);
  }
  if (node_props.isFrontierNode != 0)
  {
    boost::property_tree::ptree data_frontier;
    data_frontier.add("<xmlattr>.key", "frontier");
    data_frontier.put_value("true");
    node.add_child("data", data_frontier);
  }
  if (node_props.isEntryNode != 0)
  {
    boost::property_tree::ptree data_entry;
    data_entry.add("<xmlattr>.key", "entry");
    data_entry.put_value("true");
    node.add_child("data", data_entry);
  }
  if (!node_props.invariant.empty())
  {
    boost::property_tree::ptree data_invariant;
    data_invariant.add("<xmlattr>.key", "invariant");
    data_invariant.put_value(node_props.invariant);
    node.add_child("data", data_invariant);
  }
  if (!node_props.invariantScope.empty())
  {
    boost::property_tree::ptree data_invariant;
    data_invariant.add("<xmlattr>.key", "invariant.scope");
    data_invariant.put_value(node_props.invariantScope);
    node.add_child("data", data_invariant);
  }
}

void create_edge(boost::property_tree::ptree & edge, edge_p & edge_props,
    boost::property_tree::ptree & source, boost::property_tree::ptree & target)
{
  edge.add("<xmlattr>.id", "e" + std::to_string(edge_count++));
  edge.add("<xmlattr>.source", source.get<std::string>("<xmlattr>.id"));
  edge.add("<xmlattr>.target", target.get<std::string>("<xmlattr>.id"));
  if (!edge_props.sourcecode.empty())
  {
    boost::property_tree::ptree data_sourcecode;
    data_sourcecode.add("<xmlattr>.key", "sourcecode");
    data_sourcecode.put_value(edge_props.sourcecode);
    edge.add_child("data", data_sourcecode);
  }
  if (edge_props.startline != -1)
  {
    boost::property_tree::ptree data_lineNumberInOrigin;
    data_lineNumberInOrigin.add("<xmlattr>.key", "startline");
    data_lineNumberInOrigin.put_value(edge_props.startline);
    edge.add_child("data", data_lineNumberInOrigin);
  }
  if (edge_props.endline != -1)
  {
    boost::property_tree::ptree data_endLine;
    data_endLine.add("<xmlattr>.key", "endline");
    data_endLine.put_value(edge_props.startline);
    edge.add_child("data", data_endLine);
  }
  if (!edge_props.enterFunction.empty())
  {
    boost::property_tree::ptree data_enterFunction;
    data_enterFunction.add("<xmlattr>.key", "enterFunction");
    data_enterFunction.put_value(edge_props.enterFunction);
    edge.add_child("data", data_enterFunction);
  }
  if (!edge_props.returnFromFunction.empty())
  {
    boost::property_tree::ptree data_returnFromFunction;
    data_returnFromFunction.add("<xmlattr>.key", "returnFrom");
    data_returnFromFunction.put_value(edge_props.returnFromFunction);
    edge.add_child("data", data_returnFromFunction);
  }
  if (!edge_props.assumption.empty())
  {
    boost::property_tree::ptree data_assumption;
    data_assumption.add("<xmlattr>.key", "assumption");
    data_assumption.put_value(edge_props.assumption);
    edge.add_child("data", data_assumption);
  }
}

void create_graphml(boost::property_tree::ptree & graphml,
    std::string file_path)
{
  graphml.add("graphml.<xmlattr>.xmlns",
      "http://graphml.graphdrawing.org/xmlns");
  graphml.add("graphml.<xmlattr>.xmlns:xsi",
      "http://www.w3.org/2001/XMLSchema-instance");

  boost::property_tree::ptree key_witnessType;
  key_witnessType.add("<xmlattr>.id", "witness-type");
  key_witnessType.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "witness-type");
  key_witnessType.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "string");
  key_witnessType.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", key_witnessType);

  boost::property_tree::ptree key_assumption;
  key_assumption.add("<xmlattr>.id", "assumption");
  key_assumption.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "assumption");
  key_assumption.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "string");
  key_assumption.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", key_assumption);

  boost::property_tree::ptree key_sourcecode;
  key_sourcecode.add("<xmlattr>.id", "sourcecode");
  key_sourcecode.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "sourcecode");
  key_sourcecode.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "string");
  key_sourcecode.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", key_sourcecode);

  boost::property_tree::ptree key_sourcecodelang;
  key_sourcecodelang.add("<xmlattr>.id", "sourcecodelang");
  key_sourcecodelang.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "sourcecodeLanguage");
  key_sourcecodelang.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "string");
  key_sourcecodelang.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", key_sourcecodelang);

  boost::property_tree::ptree key_control;
  key_control.add("<xmlattr>.id", "control");
  key_control.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "control");
  key_control.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "string");
  key_control.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", key_control);

  boost::property_tree::ptree key_startline;
  key_startline.add("<xmlattr>.id", "startline");
  key_startline.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "startline");
  key_startline.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "int");
  key_startline.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", key_startline);

  boost::property_tree::ptree key_endline;
  key_endline.add("<xmlattr>.id", "endline");
  key_endline.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "endline");
  key_endline.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "int");
  key_endline.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", key_endline);

  boost::property_tree::ptree key_originfile;
  key_originfile.add("<xmlattr>.id", "originfile");
  key_originfile.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "originFileName");
  key_originfile.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "string");
  key_originfile.add("<xmlattr>.for", "edge");
  boost::property_tree::ptree key_originfile_default;
  key_originfile_default.put_value(file_path);
  key_originfile.add_child("default", key_originfile_default);
  graphml.add_child("graphml.key", key_originfile);

  boost::property_tree::ptree key_invariant;
  key_invariant.add("<xmlattr>.id", "invariant");
  key_invariant.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "invariant");
  key_invariant.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "string");
  key_invariant.add("<xmlattr>.for", "node");
  graphml.add_child("graphml.key", key_invariant);

  boost::property_tree::ptree key_invariantScope;
  key_invariantScope.add("<xmlattr>.id", "invariant.scope");
  key_invariantScope.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "invariant.scope");
  key_invariantScope.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "string");
  key_invariantScope.add("<xmlattr>.for", "node");
  graphml.add_child("graphml.key", key_invariantScope);

  boost::property_tree::ptree key_nodeType;
  key_nodeType.add("<xmlattr>.id", "nodetype");
  key_nodeType.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "nodeType");
  key_nodeType.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "string");
  key_nodeType.add("<xmlattr>.for", "node");
  boost::property_tree::ptree key_node_type_default;
  key_node_type_default.put_value("path");
  key_nodeType.add_child("default", key_node_type_default);
  graphml.add_child("graphml.key", key_nodeType);

  boost::property_tree::ptree key_frontier;
  key_frontier.add("<xmlattr>.id", "frontier");
  key_frontier.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "isFrontierNode");
  key_frontier.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "boolean");
  key_frontier.add("<xmlattr>.for", "node");
  boost::property_tree::ptree key_frontier_default;
  key_frontier_default.put_value("false");
  key_frontier.add_child("default", key_frontier_default);
  graphml.add_child("graphml.key", key_frontier);

  boost::property_tree::ptree key_violation;
  key_violation.add("<xmlattr>.id", "violation");
  key_violation.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "isViolationNode");
  key_violation.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "boolean");
  key_violation.add("<xmlattr>.for", "node");
  boost::property_tree::ptree key_violation_default;
  key_violation_default.put_value("false");
  key_violation.add_child("default", key_violation_default);
  graphml.add_child("graphml.key", key_violation);

  boost::property_tree::ptree key_entry;
  key_entry.add("<xmlattr>.id", "entry");
  key_entry.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "isEntryNode");
  key_entry.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "boolean");
  key_entry.add("<xmlattr>.for", "node");
  boost::property_tree::ptree key_entry_default;
  key_entry_default.put_value("false");
  key_entry.add_child("default", key_entry_default);
  graphml.add_child("graphml.key", key_entry);

  boost::property_tree::ptree key_sink;
  key_sink.add("<xmlattr>.id", "sink");
  key_sink.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "isSinkNode");
  key_sink.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "boolean");
  key_sink.add("<xmlattr>.for", "node");
  boost::property_tree::ptree key_sink_default;
  key_sink_default.put_value("false");
  key_sink.add_child("default", key_sink_default);
  graphml.add_child("graphml.key", key_sink);

  boost::property_tree::ptree key_enterFunction;
  key_enterFunction.add("<xmlattr>.id", "enterFunction");
  key_enterFunction.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "enterFunction");
  key_enterFunction.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "string");
  key_enterFunction.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", key_enterFunction);

  boost::property_tree::ptree key_returnFunction;
  key_returnFunction.add("<xmlattr>.id", "returnFrom");
  key_returnFunction.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
      "returnFromFunction");
  key_returnFunction.put(
      boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
      "string");
  key_returnFunction.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", key_returnFunction);

}

void create_graph(boost::property_tree::ptree & graph)
{
  graph.add("<xmlattr>.edgedefault", "directed");
  boost::property_tree::ptree data_sourcecodelang;
  data_sourcecodelang.add("<xmlattr>.key", "sourcecodelang");
  data_sourcecodelang.put_value("C");
  graph.add_child("data", data_sourcecodelang);
}
