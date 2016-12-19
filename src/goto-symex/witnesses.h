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
#include <iostream>
#include <iterator>
#include <map>
#include <string>

#include <ac_config.h>

#ifndef HAVE_OPENSSL

extern "C" {
  #include <openssl/sha.h>
}

#define SHA1_DIGEST_LENGTH 20
int generate_sha1_hash_for_file(const char * path, std::string & output)
{
  FILE * file = fopen(path, "rb");

  if(!file)
    return -1;

  unsigned char hash[SHA1_DIGEST_LENGTH];
  SHA_CTX sha1;
  SHA1_Init(&sha1);
  const int bufSize = 32768;
  char * buffer = (char *) alloca(bufSize);
  char * output_hex_hash = (char *) alloca(sizeof(char) * SHA1_DIGEST_LENGTH * 2);
  if(!buffer || !output_hex_hash)
    return -1;

  int bytesRead = 0;
  while((bytesRead = fread(buffer, 1, bufSize, file)))
	  SHA1_Update(&sha1, buffer, bytesRead);

  SHA1_Final(hash, &sha1);
  int i = 0;
  for(i = 0; i < SHA1_DIGEST_LENGTH; i++)
    sprintf(output_hex_hash + (i * 2), "%02x", hash[i]);

  output.append(output_hex_hash);
  fclose(file);
  return 0;
}

#endif /* !NO_OPENSSL */

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
  std::string invariant = "";
  std::string invariantScope = "";
} node_p;

typedef struct edge_props
{
  std::string assumption = "";
  std::string assumptionScope = "";
  std::string assumptionResultFunction = "";
  int startline = -1;
  int endline = -1;
  int startoffset = -1;
  int endoffset = -1;
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
    data_endLine.put_value(edge_props.endline);
    edge.add_child("data", data_endLine);
  }
  if (edge_props.startoffset > 0)
  {
    boost::property_tree::ptree data_startoffset;
    data_startoffset.add("<xmlattr>.key", "startoffset");
    data_startoffset.put_value(edge_props.startoffset);
    edge.add_child("data", data_startoffset);
  }
  if (edge_props.endoffset > 0)
  {
    boost::property_tree::ptree data_endoffset;
    data_endoffset.add("<xmlattr>.key", "endoffset");
    data_endoffset.put_value(edge_props.endoffset);
    edge.add_child("data", data_endoffset);
  }
  if (!edge_props.returnFromFunction.empty())
  {
    boost::property_tree::ptree data_returnFromFunction;
    data_returnFromFunction.add("<xmlattr>.key", "returnFromFunction");
    data_returnFromFunction.put_value(edge_props.returnFromFunction);
    edge.add_child("data", data_returnFromFunction);
  }
  if (!edge_props.enterFunction.empty())
  {
    boost::property_tree::ptree data_enterFunction;
    data_enterFunction.add("<xmlattr>.key", "enterFunction");
    data_enterFunction.put_value(edge_props.enterFunction);
    edge.add_child("data", data_enterFunction);
  }
  if (!edge_props.assumption.empty())
  {
    boost::property_tree::ptree data_assumption;
    data_assumption.add("<xmlattr>.key", "assumption");
    data_assumption.put_value(edge_props.assumption);
    edge.add_child("data", data_assumption);
  }
  if (!edge_props.assumptionScope.empty())
  {
    boost::property_tree::ptree data_assumptionScope;
    data_assumptionScope.add("<xmlattr>.key", "assumption.scope");
    data_assumptionScope.put_value(edge_props.assumptionScope);
    edge.add_child("data", data_assumptionScope);
  }
}

void create_graphml(boost::property_tree::ptree & graphml,
    std::string file_path)
{
  graphml.add("graphml.<xmlattr>.xmlns",
    "http://graphml.graphdrawing.org/xmlns");
  graphml.add("graphml.<xmlattr>.xmlns:xsi",
    "http://www.w3.org/2001/XMLSchema-instance");

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

  boost::property_tree::ptree key_programfile;
  key_programfile.add("<xmlattr>.id", "programfile");
  key_programfile.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
    "programfile");
  key_programfile.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
    "string");
  key_programfile.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", key_programfile);

  boost::property_tree::ptree key_programhash;
  key_programhash.add("<xmlattr>.id", "programhash");
  key_programhash.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
    "programhash");
  key_programhash.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
    "string");
  key_programhash.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", key_programhash);

  boost::property_tree::ptree key_specification;
  key_specification.add("<xmlattr>.id", "specification");
  key_specification.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
    "specification");
  key_specification.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
    "string");
  key_specification.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", key_specification);

  boost::property_tree::ptree key_memorymodel;
  key_memorymodel.add("<xmlattr>.id", "memorymodel");
  key_memorymodel.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
    "memoryModel");
  key_memorymodel.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
    "string");
  key_programhash.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", key_memorymodel);

  boost::property_tree::ptree key_architecture;
  key_architecture.add("<xmlattr>.id", "architecture");
  key_architecture.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
    "architecture");
  key_architecture.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
    "string");
  key_programhash.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", key_architecture);

  boost::property_tree::ptree key_producer;
  key_producer.add("<xmlattr>.id", "producer");
  key_producer.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
    "producer");
  key_producer.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
    "string");
  key_programhash.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", key_producer);

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

  boost::property_tree::ptree key_startoffset;
  key_startoffset.add("<xmlattr>.id", "startoffset");
  key_startoffset.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
    "startoffset");
  key_startoffset.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
    "int");
  key_startoffset.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", key_startoffset);

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

  boost::property_tree::ptree key_assumptionScope;
  key_assumptionScope.add("<xmlattr>.id", "assumption.scope");
  key_assumptionScope.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
    "assumption");
  key_assumptionScope.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
    "string");
  key_assumptionScope.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", key_assumptionScope);

  boost::property_tree::ptree key_assumption_resultFunction;
  key_assumption_resultFunction.add("<xmlattr>.id", "assumption.resultfunction");
  key_assumption_resultFunction.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
    "assumption.resultfunction");
  key_assumption_resultFunction.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
    "string");
  key_assumption_resultFunction.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", key_assumption_resultFunction);

  boost::property_tree::ptree key_assumption_scope;
  key_assumption_scope.add("<xmlattr>.id", "assumption.scope");
  key_assumption_scope.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
    "assumption.scope");
  key_assumption_scope.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
    "string");
  key_assumption_scope.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", key_assumption_scope);

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

  boost::property_tree::ptree key_returnFromFunction;
  key_returnFromFunction.add("<xmlattr>.id", "returnFromFunction");
  key_returnFromFunction.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
    "returnFromFunction");
  key_returnFromFunction.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
    "string");
  key_returnFromFunction.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", key_returnFromFunction);

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

  boost::property_tree::ptree key_endoffset;
  key_endoffset.add("<xmlattr>.id", "endoffset");
  key_endoffset.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.name", '|'),
    "endoffset");
  key_endoffset.put(
    boost::property_tree::ptree::path_type("<xmlattr>|attr.type", '|'),
    "int");
  key_endoffset.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", key_endoffset);

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
}

void create_graph(
  boost::property_tree::ptree & graph,
  std::string & filename,
  int & specification,
  const bool is_correctness)
{
  std::string hash;
  if (!filename.empty())
    generate_sha1_hash_for_file(filename.c_str(), hash);

  graph.add("<xmlattr>.edgedefault", "directed");
  boost::property_tree::ptree data_witnesstype;
  data_witnesstype.add("<xmlattr>.key", "witness-type");
  data_witnesstype.put_value(is_correctness ? "correctness_witness" : "violation_witness");
  graph.add_child("data", data_witnesstype);
  boost::property_tree::ptree data_sourcecodelang;
  data_sourcecodelang.add("<xmlattr>.key", "sourcecodelang");
  data_sourcecodelang.put_value("C");
  graph.add_child("data", data_sourcecodelang);
  boost::property_tree::ptree data_producer;
  data_producer.add("<xmlattr>.key", "producer");
  data_producer.put_value("ESBMC " + std::string(ESBMC_VERSION));
  graph.add_child("data", data_producer);
  boost::property_tree::ptree data_specification;
  data_specification.add("<xmlattr>.key", "specification");
  if (specification == 1)
    data_specification.put_value("CHECK( init(main()), LTL(G ! overflow) )");
  else if (specification == 2)
    data_specification.put_value("CHECK( init(main()), LTL(G valid-free|valid-deref|valid-memtrack) )");
  else
    data_specification.put_value("CHECK( init(main()), LTL(G ! call(__VERIFIER_error())) )");
  graph.add_child("data", data_specification);
  boost::property_tree::ptree data_programfile;
  data_programfile.add("<xmlattr>.key", "programfile");
  data_programfile.put_value(filename);
  graph.add_child("data", data_programfile);
  boost::property_tree::ptree data_programhash;
  data_programhash.add("<xmlattr>.key", "programhash");
  data_programhash.put_value(hash);
  graph.add_child("data", data_programhash);
  boost::property_tree::ptree data_memorymodel;
  data_memorymodel.add("<xmlattr>.key", "memoryModel");
  data_memorymodel.put_value("precise");
  graph.add_child("data", data_memorymodel);
  boost::property_tree::ptree data_architecture;
  data_architecture.add("<xmlattr>.key", "architecture");
  data_architecture.put_value(std::to_string(config.ansi_c.word_size) + "bit");
  graph.add_child("data", data_architecture);
}

std::string w_string_replace(
  std::string subject,
  const std::string & search,
  const std::string & replace)
{
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
     subject.replace(pos, search.length(), replace);
     pos += replace.length();
  }
  return subject;
}

int count_characters_before_line(
  const std::string & file_path,
  const int line_number,
  int & characters_in_the_line )
{
  std::string line;
  std::ifstream stream (file_path);
  int char_count = 0;
  int line_count = 0;
  characters_in_the_line = 0;
  if (stream.is_open())
  {
    while(getline(stream, line) &&
          line_count < line_number)
    {
      characters_in_the_line = line.length();
      char_count += characters_in_the_line;
      line_count++;
    }
  }
  stream.close();
  return char_count - characters_in_the_line;
}

void get_offsets_for_line_using_wc(
  const std::string & file_path,
  const int line_number,
  int & p_startoffset,
  int & p_endoffset )
{
  unsigned int startoffset = 0;
  unsigned int endoffset = 0;

  try {
    /* get the offsets */
    startoffset = std::atoi(execute_cmd("cat " + file_path + " | head -n " + std::to_string(line_number - 1) + " | wc --chars").c_str());
    endoffset = std::atoi(execute_cmd("cat " + file_path + " | head -n " + std::to_string(line_number) + " | wc --chars").c_str());
    /* count the spaces in the beginning and append to the startoffset  */
    std::string str_line = execute_cmd("cat " + file_path + " | head -n " + std::to_string(line_number) + " | tail -n 1 ");
    unsigned int i=0;
    for (i=0; i<str_line.length(); i++)
    {
      if (str_line.c_str()[i] == ' ')
        startoffset++;
      else
        break;
    }
  } catch (const std::exception& e) {
    /* nothing to do here */
  }

  p_startoffset = startoffset;
  p_endoffset = endoffset;
}

bool is_valid_witness_expr(
    const namespacet &ns,
	const irep_container<expr2t> & exp)
{
  languagest languages(ns, "C");
  std::string value;
  languages.from_expr(migrate_expr_back(exp), value);
  return (value.find("__ESBMC") &
    value.find("stdin")         &
    value.find("stdout")        &
    value.find("stderr")        &
    value.find("sys_")) == std::string::npos;
}

void get_relative_line_in_programfile(
  const std::string relative_file_path,
  const int relative_line_number,
  const std::string program_file_path,
  int & programfile_line_number)
{
  /* check if it is necessary to get the relative line */
  if (relative_file_path == program_file_path)
  {
	programfile_line_number = relative_line_number;
    return;
  }
  std::string line;
  std::string relative_content;
  std::ifstream stream_relative (relative_file_path);
  std::ifstream stream_programfile (program_file_path);
  int line_count = 0;
  /* get the relative content */
  if (stream_relative.is_open())
  {
	while(getline(stream_relative, line) &&
		  line_count < relative_line_number)
	{
	  relative_content = line;
	  line_count++;
	}
  }

  /* file for the line in the programfile */
  line_count = 1;
  if (stream_programfile.is_open())
  {
    while(getline(stream_programfile, line) &&
  	  line != relative_content)
    {
      line_count++;
    }
  }
  programfile_line_number = line_count;
}
