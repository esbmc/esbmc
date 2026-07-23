// CXL port enumeration test.
// Tests that CXL root ports and switch ports are enumerated correctly
// following the CXL 2.0 port hierarchy.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

/* CXL port types */
enum cxl_port_type {
  CXL_PORT_ROOT = 0,
  CXL_PORT_DOWNSTREAM,
  CXL_PORT_UPSTREAM,
};

/* CXL register types */
enum cxl_reg_type {
  CXL_RTYPE_CXL = 0,
  CXL_RTYPE_LC = 1,
  CXL_RTYPE_HDM_DEC = 2,
};

struct cxl_port {
  uint32_t port_type;
  uint32_t rtype;
  struct cxl_port *parent;
  struct cxl_port **children;
  uint32_t num_children;
  uint32_t max_children;
};

/* Simulated port hierarchy */
static struct cxl_port root_port;
static struct cxl_port switch_ports[4];
static int switch_count = 0;

struct cxl_port *cxl_port_create(uint32_t type, uint32_t reg_type,
                                  struct cxl_port *parent)
{
  struct cxl_port *port = NULL;

  /* Find a free slot in switch_ports */
  for (int i = 0; i < 4; i++)
  {
    if (switch_ports[i].parent == NULL)
    {
      port = &switch_ports[i];
      break;
    }
  }

  if (port == NULL)
    return NULL;

  port->port_type = type;
  port->rtype = reg_type;
  port->parent = parent;
  port->children = NULL;
  port->num_children = 0;
  port->max_children = 0;

  if (parent != NULL)
  {
    parent->children = (struct cxl_port **)0x1; /* simplified */
    parent->num_children++;
  }

  return port;
}

int cxl_enumerate_ports(void)
{
  /* Create root port */
  root_port.port_type = CXL_PORT_ROOT;
  root_port.rtype = CXL_RTYPE_CXL;
  root_port.parent = NULL;
  root_port.num_children = 0;

  /* Create switch ports connected to root */
  switch_count = 0;
  for (int i = 0; i < 3; i++)
  {
    struct cxl_port *sw = cxl_port_create(CXL_PORT_DOWNSTREAM,
                                           CXL_RTYPE_CXL, &root_port);
    if (sw == NULL)
      break;
    switch_count++;
  }

  return switch_count;
}

int main()
{
  /* Enumerate */
  int count = cxl_enumerate_ports();
  assert(count >= 0);

  /* Verify root port */
  assert(root_port.port_type == CXL_PORT_ROOT);
  assert(root_port.rtype == CXL_RTYPE_CXL);
  assert(root_port.parent == NULL);

  /* Verify switch ports are connected to root */
  for (int i = 0; i < 4; i++)
  {
    if (switch_ports[i].parent != NULL)
    {
      assert(switch_ports[i].port_type == CXL_PORT_DOWNSTREAM);
      assert(switch_ports[i].rtype == CXL_RTYPE_CXL);
      assert(switch_ports[i].parent == &root_port);
    }
  }

  /* Verify hierarchy depth: root -> switch (depth 1) */
  for (int i = 0; i < 4; i++)
  {
    if (switch_ports[i].parent != NULL)
    {
      /* Parent of parent should be NULL (root has no parent) */
      assert(switch_ports[i].parent->parent == NULL);
    }
  }
}
