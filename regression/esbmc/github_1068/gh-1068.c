//FormAI DATASET v0.1 Category: Network Topology Mapper ; Style: ephemeral


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Constants
#define MAX_NODES 100
#define MAX_LINKS 1000
#define MAX_HOPS  6

// Structs
typedef struct {
    int node_ID;
    int cost;
} Link;

typedef struct {
    int node_ID;
    Link links[MAX_NODES];
} Node;

// Global Variables
Node network[MAX_NODES];
int num_nodes = 0;
int num_links = 0;

// Forward Declarations
bool is_existing_node(int node_ID);
Node* get_node(int node_ID);
void add_node(int node_ID);
Link* get_link(int node_ID, int connected_node_ID);
void add_link(int node_ID, int connected_node_ID, int cost);
void print_topology_map();

// Main Function
int main() {

    // Create some nodes
    add_node(1);
    add_node(2);
    add_node(3);
    add_node(4);
    add_node(5);

    // Connect the nodes with links and associated costs
    add_link(1, 2, 1);
    add_link(1, 3, 2);
    add_link(2, 3, 1);
    add_link(2, 4, 1);
    add_link(3, 4, 2);
    add_link(4, 5, 1);

    // Print the topology map of the network
    print_topology_map();

    return 0;
}

// Helper Functions

// Checks if a node already exists in the network
bool is_existing_node(int node_ID) {
    for(int i = 0; i < num_nodes; i++) {
        if(network[i].node_ID == node_ID) {
            return true;
        }
    }
    return false;
}

// Returns a pointer to a given node
Node* get_node(int node_ID) {
    for(int i = 0; i < num_nodes; i++) {
        if(network[i].node_ID == node_ID) {
            return &network[i];
        }
    }
    return NULL;
}

// Adds a new node to the network
void add_node(int node_ID) {
    if(!is_existing_node(node_ID)) {
        Node new_node;
        new_node.node_ID = node_ID;
        network[num_nodes] = new_node;
        num_nodes++;
    }
}

// Returns a pointer to a link between two nodes
Link* get_link(int node_ID, int connected_node_ID) {
    Node* node = get_node(node_ID);
    if(node != NULL) {
        for(int i = 0; i < MAX_NODES; i++) {
            if(node->links[i].node_ID == connected_node_ID) {
                return &node->links[i];
            }
        }
    }
    return NULL;
}

// Adds a new link between two nodes to the network
void add_link(int node_ID, int connected_node_ID, int cost) {
    if(is_existing_node(node_ID) && is_existing_node(connected_node_ID)) {
        Link new_link;
        new_link.node_ID = connected_node_ID;
        new_link.cost = cost;
        Link* existing_link = get_link(node_ID, connected_node_ID);
        if(existing_link != NULL) {
            *existing_link = new_link;
        }
        else {
            Node* node = get_node(node_ID);
            if(node != NULL) {
                node->links[node->node_ID] = new_link;
            }
        }
        num_links++;
    }
}

// Prints the topology map of the network
void print_topology_map() {
    printf("------- Network Topology Map -------\n");
    printf("Node ID | Connected Nodes (Costs)\n");
    printf("------------------------------------\n");
    for(int i = 0; i < num_nodes; i++) {
        Node node = network[i];
        printf("%d      | ", node.node_ID);
        for(int j = 0; j < MAX_NODES; j++) {
            if(node.links[j].node_ID != 0) {
                printf("%d (%d)  ", node.links[j].node_ID, node.links[j].cost);
            }
        }
        printf("\n");
    }
    printf("------------------------------------\n");
}
