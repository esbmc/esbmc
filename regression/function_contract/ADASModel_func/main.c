/* ADASModel_func: Test function contracts with struct types
 * Tests __ESBMC_return_value with different return types (int, double)
 * in the context of an ADAS (Advanced Driver Assistance System) model
 */
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

typedef int c_frame_t;
typedef int l_frame_t;

// Global time in milliseconds
unsigned int current_time = 0;

// Camera state
typedef struct {
    c_frame_t frame;
    unsigned int last_trigger;
} Camera;

// LiDAR state
typedef struct {
    l_frame_t frame;
    unsigned int last_trigger;
} LiDAR;

// ADASProcessor state
typedef struct {
    int requestStop;
    unsigned int action_scheduled_time;
    bool action_scheduled;
    int out1;
    int out2;
} ADASProcessor;

// Brakes state
typedef struct {
    int brakesApplied;
    unsigned int brakes_applied_time;
} Brakes;

// Dashboard state
typedef struct {
    int received;
} Dashboard;

// Test functions with contracts and different return types
int get_frame_value(c_frame_t frame) {
    __ESBMC_requires(frame >= 0);
    __ESBMC_ensures(__ESBMC_return_value == frame);
    return (int)frame;
}

double calculate_delay(unsigned int time1, unsigned int time2) {
    __ESBMC_requires(time2 >= time1);
    __ESBMC_ensures(__ESBMC_return_value >= 0.0);
    __ESBMC_ensures(__ESBMC_return_value == (double)(time2 - time1));
    return (double)(time2 - time1);
}

// Camera reaction (60 fps = every 17 msec)
void camera_reaction(Camera* c, c_frame_t* out) {
    __ESBMC_requires(c != NULL);
    __ESBMC_requires(out != NULL);
    c->frame = 1;
    *out = c->frame;
}

// LiDAR reaction (30 fps = every 34 msec)
void lidar_reaction(LiDAR* l, l_frame_t* out) {
    __ESBMC_requires(l != NULL);
    __ESBMC_requires(out != NULL);
    l->frame = 2;
    *out = l->frame;
}

// ADAS Processor reaction on inputs
void adas_reaction_inputs(ADASProcessor* p, l_frame_t in1, c_frame_t in2) {
    __ESBMC_requires(p != NULL);
    __ESBMC_requires(in1 >= 0);
    __ESBMC_requires(in2 >= 0);
    // Detect danger and request stop
    p->requestStop = 1;
    p->action_scheduled = true;
    p->action_scheduled_time = current_time + 50; // Schedule after 50 msec
}

// ADAS Processor reaction on logical action
void adas_reaction_action(ADASProcessor* p) {
    __ESBMC_requires(p != NULL);
    if (p->requestStop == 1) {
        p->out1 = 1; // Signal to brakes
    }
}

// Brakes reaction (with 5 msec after delay from ADAS)
void brakes_reaction(Brakes* b, int inADAS, int inPedal) {
    __ESBMC_requires(b != NULL);
    __ESBMC_requires(inADAS >= 0 && inADAS <= 1);
    __ESBMC_requires(inPedal >= 0 && inPedal <= 1);
    if (inADAS == 1 || inPedal == 1) {
        b->brakesApplied = 1;
        b->brakes_applied_time = current_time;
    }
}

// Dashboard reaction
void dashboard_reaction(Dashboard* d, int in) {
    __ESBMC_requires(d != NULL);
    d->received = 1;
}

int main() {
    // Initialize components
    Camera camera = {0, 0};
    LiDAR lidar = {0, 0};
    ADASProcessor adas = {0, 0, false, 0, 0};
    Brakes brakes = {0, 0};
    Dashboard dashboard = {0};
    
    // Outputs from sensors
    c_frame_t camera_out = 0;
    l_frame_t lidar_out = 0;
    
    // Track important events
    unsigned int lidar_reaction_time = 0;
    bool lidar_reaction_occurred = false;
    unsigned int requestStop_time = 0;
    bool requestStop_set = false;
    
    // Simulate system execution
    // At t=0: Both Camera and LiDAR trigger
    current_time = 0;
    
    // LiDAR reaction at t=0
    lidar_reaction(&lidar, &lidar_out);
    lidar_reaction_occurred = true;
    lidar_reaction_time = current_time;
    
    // Test function with int return type and contract
    int frame_val = get_frame_value(lidar_out);
    assert(frame_val == 2);
    
    // Camera reaction at t=0
    camera_reaction(&camera, &camera_out);
    
    // ADAS receives both inputs and processes
    adas_reaction_inputs(&adas, lidar_out, camera_out);
    if (adas.requestStop == 1) {
        requestStop_set = true;
        requestStop_time = current_time;
    }
    
    // Time advances to when logical action triggers (50 msec later)
    current_time = 50;
    
    // Test function with double return type and contract
    double delay = calculate_delay(0, current_time);
    assert(delay == 50.0);
    
    if (adas.action_scheduled && current_time >= adas.action_scheduled_time) {
        adas_reaction_action(&adas);
        
        // Send to dashboard
        if (adas.out2 == 1) {
            dashboard_reaction(&dashboard, adas.out2);
        }
        
        // Send to brakes with 5 msec after delay
        current_time = 55; // 50 + 5 msec delay
        if (adas.out1 == 1) {
            brakes_reaction(&brakes, adas.out1, 0);
        }
    }
    
    // Property verification using ESBMC assertions
    if (lidar_reaction_occurred && lidar_reaction_time <= 10) {
        if (requestStop_set) {
            // Then brakes must be applied within 55ms from the start
            assert(brakes.brakesApplied == 1 && "Brakes should be applied");
            assert(brakes.brakes_applied_time <= 55 && "Brakes should be applied within 55ms");
        }
    }
    
    // Additional verification assertions
    assert(lidar_reaction_occurred && "LiDAR reaction should occur");
    assert(requestStop_set && "Request stop should be set");
    assert(adas.requestStop == 1 && "ADAS requestStop should be 1");
    assert(brakes.brakesApplied == 1 && "Brakes should be applied");
    
    return 0;
}

