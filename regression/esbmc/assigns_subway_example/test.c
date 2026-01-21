// Simplified Subway model without struct fields in assigns
// This tests that assigns prevents false counterexamples due to over-approximation

int global_time = 0;
int ums_done_received = 0;
int ums_done_time = -1;
int ums_inUse = 0;

void ums_init() {
    __ESBMC_assigns("ums_inUse");
    __ESBMC_ensures(ums_inUse == 0);
    
    ums_inUse = 0;
}

void ums_done() {
    __ESBMC_assigns("ums_inUse", "ums_done_received", "ums_done_time");
    
    __ESBMC_ensures(ums_inUse == 0);
    __ESBMC_ensures(ums_done_received == 1);
    __ESBMC_ensures(ums_done_time == __ESBMC_old(global_time));
    
    ums_inUse = 0;
    ums_done_received = 1;
    ums_done_time = global_time;
}

int main() {
    // Initialize
    ums_init();
    global_time = 0;
    
    // BUG: Wait 2 minutes before using the system
    global_time += 2;
    
    // Use system (takes 10 minutes)
    global_time += 10;
    
    // Send done signal
    ums_done();
    
    // Property: done should be received AFTER 11 minutes
    // With bug: done is received at minute 12
    // Expected: Verification should SUCCEED (property holds)
    if (ums_done_received) {
        __ESBMC_assert(ums_done_time > 11, 
                       "done_received_after_11_minutes");
    }
    
    return 0;
}
