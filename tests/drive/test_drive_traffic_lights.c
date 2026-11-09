#include "include/drive_fixture.h"
#include "include/test.h"

// Three lights of one junction (phases 0..2) plus one standalone light; eval cycle 10/3/2 s at dt 0.1.
#define LIGHT_COUNT 4
#define STATE_COUNT 900
#define SLOT_STEPS 150

static int light_states[LIGHT_COUNT][STATE_COUNT];

static Drive drive_test_lights_env(TrafficControlElement *lights, int eval_mode) {
    Drive env = {0};
    env.eval_mode = eval_mode;
    env.traffic_light_junction_phases = 1;
    env.dt = 0.1f;
    env.scenario_length = STATE_COUNT;
    env.num_traffic_elements = LIGHT_COUNT;
    env.traffic_elements = lights;
    rng_seed(&env.rng_state, 7);
    for (int i = 0; i < LIGHT_COUNT; i++) {
        lights[i] = (TrafficControlElement) {0};
        lights[i].type = TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
        lights[i].state_size = STATE_COUNT;
        lights[i].states = light_states[i];
        lights[i].junction_id = i < 3 ? 42 : -1;
        lights[i].phase_idx = i < 3 ? i : -1;
    }
    return env;
}

static int count_state(const int *states, int state) {
    int count = 0;
    for (int t = 0; t < STATE_COUNT; t++) {
        count += states[t] == state;
    }
    return count;
}

static int test_junction_lights_never_green_together(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = drive_test_lights_env(lights, 1);
    generate_traffic_light_states(&env);
    for (int t = 0; t < STATE_COUNT; t++) {
        int non_red = 0;
        for (int i = 0; i < 3; i++) {
            non_red += lights[i].states[t] != TRAFFIC_CONTROL_STATE_RED;
        }
        EXPECT_TRUE(non_red <= 1);
    }
    return 0;
}

static int test_junction_lights_share_cycle_and_take_turns(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = drive_test_lights_env(lights, 1);
    generate_traffic_light_states(&env);
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_GREEN), 200);
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_YELLOW), 60);
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_RED), 640);
    }
    for (int t = 0; t < STATE_COUNT - SLOT_STEPS; t++) {
        // Green slot of phase k is followed one slot later by the green slot of phase (k+1) % 3.
        for (int i = 0; i < 3; i++) {
            if (lights[i].states[t] != TRAFFIC_CONTROL_STATE_GREEN) {
                continue;
            }
            EXPECT_EQ_INT(lights[(i + 1) % 3].states[t + SLOT_STEPS], TRAFFIC_CONTROL_STATE_GREEN);
        }
    }
    return 0;
}

static int test_standalone_light_keeps_single_phase_cycle(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = drive_test_lights_env(lights, 1);
    generate_traffic_light_states(&env);
    EXPECT_EQ_INT(count_state(lights[3].states, TRAFFIC_CONTROL_STATE_GREEN), 600);
    EXPECT_EQ_INT(count_state(lights[3].states, TRAFFIC_CONTROL_STATE_YELLOW), 180);
    EXPECT_EQ_INT(count_state(lights[3].states, TRAFFIC_CONTROL_STATE_RED), 120);
    for (int t = 0; t < STATE_COUNT - SLOT_STEPS; t++) {
        EXPECT_EQ_INT(lights[3].states[t + SLOT_STEPS], lights[3].states[t]);
    }
    return 0;
}

static int test_junction_phases_disabled_makes_every_light_standalone(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = drive_test_lights_env(lights, 1);
    env.traffic_light_junction_phases = 0;
    generate_traffic_light_states(&env);
    for (int i = 0; i < LIGHT_COUNT; i++) {
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_GREEN), 600);
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_YELLOW), 180);
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_RED), 120);
    }
    return 0;
}

static int test_training_lights_stay_exclusive_unless_removed(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    int exclusive_violations = 0;
    for (int seed = 0; seed < 20; seed++) {
        Drive env = drive_test_lights_env(lights, 0);
        rng_seed(&env.rng_state, (uint64_t) seed);
        generate_traffic_light_states(&env);
        for (int t = 0; t < STATE_COUNT; t++) {
            int cycling_non_red = 0;
            for (int i = 0; i < 3; i++) {
                int state = lights[i].states[t];
                int is_cycling = count_state(lights[i].states, TRAFFIC_CONTROL_STATE_RED) > 0;
                cycling_non_red += is_cycling && state != TRAFFIC_CONTROL_STATE_RED;
            }
            exclusive_violations += cycling_non_red > 1;
        }
    }
    EXPECT_EQ_INT(exclusive_violations, 0);
    return 0;
}

// Junction fully off while the standalone light is not: individual removal alone needs three 20% draws to line up.
static int count_junction_only_off_episodes(int junction_phases, int seed_count) {
    TrafficControlElement lights[LIGHT_COUNT];
    int junction_only_off_count = 0;
    for (int seed = 0; seed < seed_count; seed++) {
        Drive env = drive_test_lights_env(lights, 0);
        env.traffic_light_junction_phases = junction_phases;
        rng_seed(&env.rng_state, (uint64_t) seed);
        generate_traffic_light_states(&env);
        int off_lights = 0;
        for (int i = 0; i < LIGHT_COUNT; i++) {
            off_lights += count_state(lights[i].states, TRAFFIC_CONTROL_STATE_OFF) == STATE_COUNT;
        }
        int standalone_off = count_state(lights[3].states, TRAFFIC_CONTROL_STATE_OFF) == STATE_COUNT;
        junction_only_off_count += off_lights == 3 && !standalone_off;
    }
    return junction_only_off_count;
}

// Expected rate per episode: 0.8 (no episode disable) * ~0.206 (group removed) * 0.8 (standalone kept) ~= 13%.
static int test_training_group_removal_turns_off_whole_junction(void) {
    for (int junction_phases = 0; junction_phases <= 1; junction_phases++) {
        int junction_only_off_count = count_junction_only_off_episodes(junction_phases, 500);
        EXPECT_TRUE(junction_only_off_count >= 40);
        EXPECT_TRUE(junction_only_off_count <= 100);
    }
    return 0;
}

static int test_eval_never_removes_lights(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    for (int seed = 0; seed < 50; seed++) {
        Drive env = drive_test_lights_env(lights, 1);
        rng_seed(&env.rng_state, (uint64_t) seed);
        generate_traffic_light_states(&env);
        for (int i = 0; i < LIGHT_COUNT; i++) {
            EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_OFF), 0);
        }
    }
    return 0;
}

// Gigaflow schedules include the terminal step, so the bootstrap observation reads a real light state
static int test_schedule_covers_terminal_observation(void) {
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.eval_mode = 1; // default CARLA cycles: no light is removed, so the terminal state is never OFF
    allocate(&env);
    c_reset(&env);
    int light_count = 0;
    TrafficControlElement *probe_light = NULL;
    for (int i = 0; i < env.num_traffic_elements; i++) {
        TrafficControlElement *tc = &env.traffic_elements[i];
        if (tc->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        EXPECT_EQ_INT(tc->state_size, env.scenario_length + 1);
        EXPECT_TRUE(tc->states[env.scenario_length] != TRAFFIC_CONTROL_STATE_OFF);
        probe_light = tc;
        light_count++;
    }
    EXPECT_TRUE(light_count > 0);
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->sim_x = (probe_light->stop_line[0] + probe_light->stop_line[3]) * 0.5f;
    agent->sim_y = (probe_light->stop_line[1] + probe_light->stop_line[4]) * 0.5f;
    agent->sim_z = (probe_light->stop_line[2] + probe_light->stop_line[5]) * 0.5f;
    copy_pose_to_prev(agent);
    env.timestep = env.scenario_length;
    compute_observations(&env);
    int valid_base = compute_observation_size(&env) - OBS_VALID_COUNT_FEATURES;
    int traffic_base = valid_base - env.obs_slots_traffic_controls_n * TRAFFIC_CONTROL_FEATURES;
    EXPECT_TRUE(env.observations[valid_base + 3] >= 1.0f);
    int observed_state = (int) env.observations[traffic_base + TRAFFIC_CONTROL_FEATURES - 1];
    EXPECT_EQ_INT(observed_state, probe_light->states[env.scenario_length]);
    free_allocated(&env);
    return 0;
}

// --- K3 truth-timing knob: bin-provided states (replay semantics) vs the procedural fill ---

#define TRUTH_BIN_STATES 40
static int truth_bins[LIGHT_COUNT][STATE_COUNT];

static void truth_fill_bin(int *bin, int size, int phase) {
    for (int t = 0; t < size; t++) {
        bin[t] = ((t / 10 + phase) % 2) ? TRAFFIC_CONTROL_STATE_GREEN : TRAFFIC_CONTROL_STATE_RED;
    }
}

static Drive truth_test_env(TrafficControlElement *lights, int truth_on) {
    Drive env = drive_test_lights_env(lights, 1);
    env.traffic_light_truth_timing = truth_on;
    for (int i = 0; i < LIGHT_COUNT; i++) {
        truth_fill_bin(truth_bins[i], i == 0 ? TRUTH_BIN_STATES : STATE_COUNT, i);
        lights[i].bin_states = truth_bins[i];
        lights[i].bin_state_size = i == 0 ? TRUTH_BIN_STATES : STATE_COUNT;
    }
    return env;
}

// Knob on: every light keeps its bin program; the short bin repeats its last state past the bin
// length, and no light is counted as a procedural fallback.
static int test_truth_timing_preserves_bin_states(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = truth_test_env(lights, 1);
    generate_traffic_light_states(&env);
    EXPECT_EQ_INT(env.truth_timing_fallback_lights, 0);
    for (int i = 0; i < LIGHT_COUNT; i++) {
        int bin_size = lights[i].bin_state_size;
        for (int t = 0; t < STATE_COUNT; t++) {
            int src = t < bin_size ? t : bin_size - 1;
            EXPECT_EQ_INT(lights[i].states[t], truth_bins[i][src]);
        }
    }
    return 0;
}

// Knob off (legacy): bin copies are ignored and the procedural fill overwrites the programs.
static int test_truth_timing_off_overwrites_bin_states(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = truth_test_env(lights, 0);
    generate_traffic_light_states(&env);
    int differing = 0;
    for (int i = 0; i < LIGHT_COUNT; i++) {
        for (int t = 0; t < STATE_COUNT; t++) {
            differing += lights[i].states[t] != truth_bins[i][t % lights[i].bin_state_size];
        }
    }
    EXPECT_TRUE(differing > 0);
    return 0;
}

// Attaching a pristine bin copy alone must not disturb the legacy knob-off stream: same seed,
// bin-free lights and bin-carrying lights fill identically.
static int test_truth_timing_off_matches_bin_free_env(void) {
    TrafficControlElement with_bins[LIGHT_COUNT];
    TrafficControlElement without_bins[LIGHT_COUNT];
    Drive a = truth_test_env(with_bins, 0);
    Drive b = drive_test_lights_env(without_bins, 1);
    generate_traffic_light_states(&a);
    generate_traffic_light_states(&b);
    EXPECT_EQ_INT(a.truth_timing_fallback_lights, 0);
    EXPECT_EQ_INT(b.truth_timing_fallback_lights, 0);
    for (int i = 0; i < LIGHT_COUNT; i++) {
        EXPECT_TRUE(memcmp(with_bins[i].states, without_bins[i].states, STATE_COUNT * sizeof(int)) == 0);
    }
    return 0;
}

// Knob on with a geometry-only light (CARLA-style: no bin states) falls back to the procedural
// fill for that light and records it in truth_timing_fallback_lights; bin-carrying junction
// lights stay preserved.
static int test_truth_timing_fallback_counts_geometry_only_light(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = truth_test_env(lights, 1);
    lights[3].bin_states = NULL;
    lights[3].bin_state_size = 0;
    generate_traffic_light_states(&env);
    EXPECT_EQ_INT(env.truth_timing_fallback_lights, 1);
    for (int t = 0; t < STATE_COUNT; t++) {
        EXPECT_EQ_INT(lights[3].states[t] >= TRAFFIC_CONTROL_STATE_RED
                          && lights[3].states[t] <= TRAFFIC_CONTROL_STATE_OFF,
                      1);
        int src = t < TRUTH_BIN_STATES ? t : TRUTH_BIN_STATES - 1;
        EXPECT_EQ_INT(lights[0].states[t], truth_bins[0][src]);
    }
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_junction_lights_never_green_together);
    RUN_TEST(test_junction_lights_share_cycle_and_take_turns);
    RUN_TEST(test_standalone_light_keeps_single_phase_cycle);
    RUN_TEST(test_junction_phases_disabled_makes_every_light_standalone);
    RUN_TEST(test_training_lights_stay_exclusive_unless_removed);
    RUN_TEST(test_training_group_removal_turns_off_whole_junction);
    RUN_TEST(test_eval_never_removes_lights);
    RUN_TEST(test_schedule_covers_terminal_observation);
    RUN_TEST(test_truth_timing_preserves_bin_states);
    RUN_TEST(test_truth_timing_off_overwrites_bin_states);
    RUN_TEST(test_truth_timing_off_matches_bin_free_env);
    RUN_TEST(test_truth_timing_fallback_counts_geometry_only_light);
    return test_summary(failures);
}
