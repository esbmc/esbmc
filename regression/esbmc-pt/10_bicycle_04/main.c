#include <time.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>

#include <sys/time.h>

enum statet {
	trip_state = 0, speed_state = 1, total_state = 2, time_state = 3
};

bool end_threads = false;
pthread_mutex_t cycle_dist_lock;
uint64_t cycle_distance_m = 0;
uint64_t total_cycle_distance_m = 0;
enum statet cur_state = trip_state;
struct timeval starttime;

static unsigned int
state2time(enum statet thestate)
{

	switch (thestate) {
	case trip_state:
		return 200;
	case speed_state:
		return 100;
	case total_state:
		return 500;
	case time_state:
		return 1000;
	}
}

void *
printing_thread(void *dummy)
{
	enum statet captured_state;
	uint64_t captured_distance, s_since, captured_total_distance;
	struct timeval captured_time, now;
	struct timespec time_to_sleep;
	double speed;

	while (!end_threads) {

		pthread_mutex_lock(&cycle_dist_lock);
		captured_state = cur_state;
		captured_distance = cycle_distance_m;
		captured_total_distance = total_cycle_distance_m;
		captured_time = starttime;
		pthread_mutex_unlock(&cycle_dist_lock);

		gettimeofday(&now, NULL);

		switch (cur_state) {
		case trip_state:
			/* Mileage */
			fprintf(stderr, "Mileage: %llum\n", captured_distance);
			break;
		case speed_state:
			s_since = now.tv_sec - captured_time.tv_sec;
			if (s_since != 0)
				speed = (double)captured_distance / (double)s_since;
			else
				speed = 0;

			fprintf(stderr, "Speed: %f M/S\n", speed);
			break;
		case total_state:
			fprintf(stderr, "Total Mileage: %llum\n",
					captured_total_distance);
			break;
		case time_state:
			s_since = now.tv_sec - captured_time.tv_sec;
			fprintf(stderr, "Time: %d seconds\n", s_since);
			break;
		}

		time_to_sleep.tv_nsec = state2time(captured_state) * 1000000;
		time_to_sleep.tv_sec = time_to_sleep.tv_nsec / 1000000000;
		time_to_sleep.tv_nsec %= 1000000000;
		nanosleep(&time_to_sleep, NULL);
	}

	return NULL;
}

void *
cycling_thread(void *dummy)
{
	struct timespec time_to_sleep;

	time_to_sleep.tv_sec = 0;
	time_to_sleep.tv_nsec = 100000000;

	// Follow existing progress formula, not defined in spec

	while (!end_threads) {
		nanosleep(&time_to_sleep, NULL);
		if ((rand() % 3) == 0) {
			pthread_mutex_lock(&cycle_dist_lock);
			cycle_distance_m++;
			total_cycle_distance_m++;
			pthread_mutex_unlock(&cycle_dist_lock);
		}
	}

	return NULL;
}

int
main()
{
	int input;
	pthread_t cycling, printing;

	gettimeofday(&starttime, NULL);
	pthread_mutex_init(&cycle_dist_lock, NULL);
	pthread_create(&cycling, NULL, cycling_thread, NULL);
	pthread_create(&printing, NULL, printing_thread, NULL);

	do {
		printf("Cycling options:\n");
		printf("1) Reset button\n");
		printf("2) Mode button\n");
		printf("3) Quit\n");
		input = 0;
		scanf("%d", &input);

		switch (input) {
		default:
			printf("Not a valid input\n");
			break;
		case 1:
			pthread_mutex_lock(&cycle_dist_lock);
			gettimeofday(&starttime, NULL);
			cycle_distance_m = 0;
			pthread_mutex_unlock(&cycle_dist_lock);
			break;
		case 2:
			pthread_mutex_lock(&cycle_dist_lock);
			cur_state = (cur_state + 1) % 4;
			pthread_mutex_unlock(&cycle_dist_lock);
			break;
		case 3:
			goto out; // 100% legitimate use of goto
		}
	} while (1);

out:
	// Clear up
	end_threads = true;
	pthread_join(cycling, NULL);
	pthread_join(printing, NULL);

	return 0;
}
