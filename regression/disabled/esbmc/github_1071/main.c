//FormAI DATASET v0.1 Category: Planet Gravity Simulation ; Style: retro
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#define G 6.67430e-11 // gravitational constant
#define MASS_RANGE 10 // mass range of planets
#define MIN_DISTANCE 50 // minimum distance between planets
#define MAX_DISTANCE 200 // maximum distance between planets
#define GRAVITY_STEPS 100 // number of steps for gravity calculation
#define SCREEN_WIDTH 80 // width of the screen
#define SCREEN_HEIGHT 20 // height of the screen

struct planet {
    double x; // x-coordinate
    double y; // y-coordinate
    double vx; // x-component of velocity
    double vy; // y-component of velocity
    double mass; // mass of the planet
    char symbol; // symbol representing the planet
};

void clear_screen() {
    system("clear");
}

void wait(int milliseconds) {
    usleep(milliseconds * 1000);
}

double rand_double(double min, double max) {
    return min + ((double) rand() / RAND_MAX) * (max - min);
}

char random_symbol() {
    char symbols[] = { '.', '*', 'o', '+' };
    int index = rand() % sizeof(symbols);
    return symbols[index];
}

double distance(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return sqrt(dx * dx + dy * dy);
}

double force(double m1, double m2, double d) {
    return G * m1 * m2 / (d * d);
}

double acceleration(double f, double m) {
    return f / m;
}

double velocity(double v, double a, double t) {
    return v + a * t;
}

double position(double p, double v, double t) {
    return p + v * t;
}

void update_position(struct planet *p, double t) {
    p->x = position(p->x, p->vx, t);
    p->y = position(p->y, p->vy, t);
}

void update_velocity(struct planet *p, double ax, double ay, double t) {
    p->vx = velocity(p->vx, ax, t);
    p->vy = velocity(p->vy, ay, t);
}

void print_planet(struct planet p) {
    int x = (int) p.x;
    int y = (int) p.y;
    if (x >= 0 && x < SCREEN_WIDTH && y >= 0 && y < SCREEN_HEIGHT) {
        printf("\033[%d;%dH%c", y + 1, x + 1, p.symbol);
    }
}

void draw_planets(struct planet *planets, int num_planets) {
    clear_screen();
    for (int i = 0; i < num_planets; i++) {
        print_planet(planets[i]);
    }
    fflush(stdout);
}

void init_planets(struct planet *planets, int num_planets) {
    srand(time(NULL));
    for (int i = 0; i < num_planets; i++) {
        planets[i].x = rand_double(0, SCREEN_WIDTH);
        planets[i].y = rand_double(0, SCREEN_HEIGHT);
        planets[i].vx = 0;
        planets[i].vy = 0;
        planets[i].mass = rand_double(1, MASS_RANGE);
        planets[i].symbol = random_symbol();
        for (int j = 0; j < i; j++) {
            double d = distance(planets[i].x, planets[i].y, planets[j].x, planets[j].y);
            if (d < MIN_DISTANCE) {
                i--;
                break;
            }
        }
    }
}

void calculate_gravity(struct planet *planets, int num_planets, double t) {
    double ax[num_planets], ay[num_planets];
    for (int i = 0; i < num_planets; i++) {
        ax[i] = 0;
        ay[i] = 0;
        for (int j = 0; j < num_planets; j++) {
            if (i != j) {
                double d = distance(planets[i].x, planets[i].y, planets[j].x, planets[j].y);
                double f = force(planets[i].mass, planets[j].mass, d);
                double axij = acceleration(f, planets[i].mass) * (planets[j].x - planets[i].x) / d;
                double ayij = acceleration(f, planets[i].mass) * (planets[j].y - planets[i].y) / d;
                ax[i] += axij;
                ay[i] += ayij;
            }
        }
    }
    for (int i = 0; i < num_planets; i++) {
        update_velocity(&planets[i], ax[i], ay[i], t);
        update_position(&planets[i], t);
    }
}

int main() {
    struct planet planets[4];
    init_planets(planets, 4);
    for (int i = 0; i < 1000; i++) {
        draw_planets(planets, 4);
        calculate_gravity(planets, 4, 0.1);
        wait(50);
    }
    return 0;
}
