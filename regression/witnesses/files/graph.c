#include<iostream>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/graphml.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <string>

typedef struct graph_props {
   std::string sourcecodeLanguage;
} graph_p;

typedef struct node_props {
   std::string nodeType;
   bool isFrontierNode;
   bool isViolationNode;
   bool isEntryNode;
   bool isSinkNode;
} node_p;

typedef struct edge_props { 
   std::string assumption;
   std::string sourcecode; 
   std::string tokenSet;
   std::string originTokenSet;
   std::string negativeCase;
   int lineNumberInOrigin;
   std::string originFileName;
   std::string enterFunction;
   std::string returnFromFunction;
} edge_p;

void create_node(boost::property_tree::ptree & node, node_p & node_props){  
  node.add("<xmlattr>.id", "n1");
  if (node_props.nodeType != ""){
     boost::property_tree::ptree data_nodetype;
     data_nodetype.add("<xmlattr>.key", "notetype");
     data_nodetype.put_value(node_props.nodeType);  
     node.add_child("data",data_nodetype);
  }
  if (node_props.isViolationNode != 0){
     boost::property_tree::ptree data_violation;
     data_violation.add("<xmlattr>.key", "violation");
     data_violation.put_value("true");  
     node.add_child("data",data_violation);
  }
  if (node_props.isSinkNode != 0){
     boost::property_tree::ptree data_sink;
     data_sink.add("<xmlattr>.key", "sink");
     data_sink.put_value("true");  
     node.add_child("data",data_sink);
  }
  if (node_props.isFrontierNode != 0){
     boost::property_tree::ptree data_frontier;
     data_frontier.add("<xmlattr>.key", "frontier");
     data_frontier.put_value("true"); 
     node.add_child("data",data_frontier); 
  }
  if (node_props.isEntryNode != 0){
     boost::property_tree::ptree data_entry;
     data_entry.add("<xmlattr>.key", "entry");
     data_entry.put_value("true");  
     node.add_child("data",data_entry);
  }
}

void create_edge(boost::property_tree::ptree & edge, edge_p & edge_props, boost::property_tree::ptree & source, boost::property_tree::ptree & target){  
  edge.add("<xmlattr>.id", "n1");
  edge.add("<xmlattr>.source", source.get<std::string>("<xmlattr>.id"));
  edge.add("<xmlattr>.target", target.get<std::string>("<xmlattr>.id"));
  if (edge_props.assumption != ""){
     boost::property_tree::ptree data_assumption;
     data_assumption.add("<xmlattr>.key", "assumption");
     data_assumption.put_value(edge_props.assumption);  
     edge.add_child("data", data_assumption);
  }
  if (edge_props.enterFunction != ""){
     boost::property_tree::ptree data_enterFunction;
     data_enterFunction.add("<xmlattr>.key", "enterFunction");
     data_enterFunction.put_value(edge_props.enterFunction);  
     edge.add_child("data",data_enterFunction);
  }
  if (edge_props.lineNumberInOrigin != -1){
     boost::property_tree::ptree data_lineNumberInOrigin;
     data_lineNumberInOrigin.add("<xmlattr>.key", "originline");
     data_lineNumberInOrigin.put_value(edge_props.lineNumberInOrigin);  
     edge.add_child("data", data_lineNumberInOrigin);
  } 
  if (edge_props.negativeCase != ""){
     boost::property_tree::ptree data_negativeCase;
     data_negativeCase.add("<xmlattr>.key", "negated");
     data_negativeCase.put_value(edge_props.negativeCase);  
     edge.add_child("data", data_negativeCase);
  } 
  if (edge_props.originFileName != ""){
     boost::property_tree::ptree data_originFileName;
     data_originFileName.add("<xmlattr>.key", "originfile");
     data_originFileName.put_value(edge_props.originFileName);  
     edge.add_child("data", data_originFileName);
  } 
  if (edge_props.originTokenSet != ""){
     boost::property_tree::ptree data_originTokenSet;
     data_originTokenSet.add("<xmlattr>.key", "origintokens");
     data_originTokenSet.put_value(edge_props.originTokenSet);  
     edge.add_child("data", data_originTokenSet);
  }
  if (edge_props.returnFromFunction != ""){
     boost::property_tree::ptree data_returnFromFunction;
     data_returnFromFunction.add("<xmlattr>.key", "returnFrom");
     data_returnFromFunction.put_value(edge_props.returnFromFunction);  
     edge.add_child("data", data_returnFromFunction);
  }
  if (edge_props.sourcecode != ""){
     boost::property_tree::ptree data_sourcecode;
     data_sourcecode.add("<xmlattr>.key", "sourcecode");
     data_sourcecode.put_value(edge_props.sourcecode);  
     edge.add_child("data", data_sourcecode);
  }
  if (edge_props.tokenSet != ""){
     boost::property_tree::ptree data_tokenSet;
     data_tokenSet.add("<xmlattr>.key", "tokens");
     data_tokenSet.put_value(edge_props.tokenSet);  
     edge.add_child("data", data_tokenSet);
  }
}

int main(int, char *[]) {
/*
   typedef boost::adjacency_list <boost::listS, boost::vecS, boost::directedS, node_p, edge_p, graph_p> Graph;
   typedef boost::graph_traits<Graph>::vertex_descriptor node_t;
   typedef boost::graph_traits<Graph>::edge_descriptor edge_t;
     
   Graph g;

   boost::dynamic_properties dp;
   
   dp.property("nodeType", boost::get(&node_p::nodeType, g));
   dp.property("isFrontierNode", boost::get(&node_p::isFrontierNode, g));
   dp.property("isViolationNode", boost::get(&node_p::isViolationNode, g));
   dp.property("isEntryNode", boost::get(&node_p::isEntryNode, g));
   dp.property("isSinkNode", boost::get(&node_p::isSinkNode, g)); 
  
   dp.property("assumption", boost::get(&edge_p::assumption, g));
   dp.property("sourcecode", boost::get(&edge_p::sourcecode, g));
   dp.property("tokenSet", boost::get(&edge_p::tokenSet, g));
   dp.property("originTokenSet", boost::get(&edge_p::originTokenSet, g));
   dp.property("negativeCase", boost::get(&edge_p::negativeCase, g));
   dp.property("lineNumberInOrigin", boost::get(&edge_p::lineNumberInOrigin, g));
   dp.property("originFileName", boost::get(&edge_p::enterFunction, g));
   dp.property("returnFromFunction", boost::get(&edge_p::returnFromFunction, g));
  
   node_t u = boost::add_vertex(g);
   node_t v = boost::add_vertex(g);

   edge_t e; bool b;
   boost::tie(e,b) = boost::add_edge(u,v,g);

   std::ofstream graphmlOutFile("graph.xml");
   boost::write_graphml(graphmlOutFile, g, dp, false);
   graphmlOutFile.close();
 */
   /* adjust xml */

  boost::property_tree::ptree graph;

  boost::property_tree::ptree node;
 
  node_p props;
  props.isSinkNode = 1;
  props.isEntryNode = 1;
  create_node(node, props);

  edge_p e_props;
  e_props.assumption = "a=1";
  boost::property_tree::ptree edge;
  create_edge(edge, e_props, node, node);

  graph.add_child("graph.node", node);
  graph.add_child("graph.edge", edge);

  boost::property_tree::xml_writer_settings<char> settings('\t', 1);;
  boost::property_tree::write_xml("graph2.xml", graph, std::locale(), settings);

}
