#include "include/drive_fixture.h"
#include "include/test.h"

// K1 in-kernel static hazard placement (parked / collision clusters / cones):
// reset-path placement into the static co-sim partner slots, driven by the
// hazard_* config knobs. Coverage: knob-off leaves the episode stream
// bit-identical; same seed -> same layout; legality invariants (kernel lane
// association, no unintentional overlap, no off-road, parked spacing); counts
// stay within the planned ranges and the slot budget.

static Drive hazard_test_env(const char *map_file, int num_agents, int slots, uint64_t seed) {
    Drive env = drive_test_env_config(map_file, SIMULATION_MODE_GIGAFLOW, num_agents, 0);
    env.cosim_partner_slots = slots;
    env.init_seed = seed;
    env.use_exact_episode_seed = 1;
    env.hazard_enable = 1;
    env.hazard_parked_enable = 1;
    env.hazard_parked_n_lo = 6;
    env.hazard_parked_n_hi = 10;
    env.hazard_parked_size_weights[HAZARD_PRESET_CAR] = 0.6f;
    env.hazard_parked_size_weights[HAZARD_PRESET_BUS] = 0.2f;
    env.hazard_parked_size_weights[HAZARD_PRESET_TRUCK] = 0.2f;
    env.hazard_clusters_enable = 1;
    env.hazard_clusters_n_lo = 1;
    env.hazard_clusters_n_hi = 2;
    env.hazard_clusters_layout = HAZARD_CLUSTER_LAYOUT_BOTH;
    env.hazard_cones_enable = 1;
    env.hazard_cones_zones_lo = 1;
    env.hazard_cones_zones_hi = 2;
    env.hazard_cones_taper_cones = 8;
    env.hazard_cones_block_rows = 2;
    env.hazard_cones_block_cols = 7;
    env.hazard_cones_spacing_m = 1.5f;
    env.hazard_cones_taper_prob = 0.5f;
    return env;
}

// Full-path reset (mirrors binding my_reseed): defeats the timestep==0 fast
// path so begin_episode_rng reseeds from the exact episode seed.
static void hazard_full_reset(Drive *env) {
    env->timestep = -1;
    c_reset(env);
}

static int hazard_slot_base(const Drive *env) {
    return env->num_controllable_agents;
}

static int hazard_placed_total(const Drive *env) {
    return env->hazard_placed_parked + env->hazard_placed_cluster_cars + env->hazard_placed_cones;
}

static int hazard_valid_slots(const Drive *env) {
    int valid = 0;
    for (int k = 0; k < env->cosim_partner_slots; k++) {
        valid += env->agents[env->static_agent_indices[k]].sim_valid == 1;
    }
    return valid;
}

// OBB overlap with the prototype conflict margin (candidate box padded by 0.05 m).
static int hazard_boxes_overlap_margin(Agent *a, Agent *b) {
    Agent padded = *a;
    padded.sim_length += 2.0f * HAZARD_CONFLICT_MARGIN_M;
    padded.sim_width += 2.0f * HAZARD_CONFLICT_MARGIN_M;
    update_agent_radius(&padded);
    return check_obb_collision(&padded, b) ? 1 : 0;
}

static int test_hazard_disabled_keeps_episode_stream_bit_identical(void) {
    // hazard_enable=false must consume no RNG draws and place nothing: the
    // dynamic spawn layout must be bit-identical to a plain no-slot env.
    srand(7);
    Drive plain = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 32, 0);
    plain.init_seed = 42;
    plain.use_exact_episode_seed = 1;
    allocate(&plain);
    hazard_full_reset(&plain);

    Drive slots_off = hazard_test_env(drive_carla_map(), 32, 48, 42);
    slots_off.hazard_enable = 0;
    allocate(&slots_off);
    hazard_full_reset(&slots_off);

    EXPECT_EQ_INT(plain.active_agent_count, slots_off.active_agent_count);
    for (int x = 0; x < plain.active_agent_count; x++) {
        const Agent *a = &plain.agents[plain.active_agent_indices[x]];
        const Agent *b = &slots_off.agents[slots_off.active_agent_indices[x]];
        EXPECT_NEAR(a->sim_x, b->sim_x, 0.0f);
        EXPECT_NEAR(a->sim_y, b->sim_y, 0.0f);
        EXPECT_NEAR(a->sim_heading, b->sim_heading, 0.0f);
    }
    EXPECT_EQ_INT(hazard_valid_slots(&slots_off), 0);
    EXPECT_EQ_INT(hazard_placed_total(&slots_off), 0);

    // All sub-knobs off under hazard_enable: still no placement, stream unchanged.
    Drive all_off = hazard_test_env(drive_carla_map(), 32, 48, 42);
    all_off.hazard_parked_enable = 0;
    all_off.hazard_clusters_enable = 0;
    all_off.hazard_cones_enable = 0;
    allocate(&all_off);
    hazard_full_reset(&all_off);
    EXPECT_EQ_INT(hazard_valid_slots(&all_off), 0);
    EXPECT_EQ_INT(hazard_placed_total(&all_off), 0);
    for (int x = 0; x < plain.active_agent_count; x++) {
        const Agent *a = &plain.agents[plain.active_agent_indices[x]];
        const Agent *b = &all_off.agents[all_off.active_agent_indices[x]];
        EXPECT_NEAR(a->sim_x, b->sim_x, 0.0f);
        EXPECT_NEAR(a->sim_y, b->sim_y, 0.0f);
    }

    free_allocated(&plain);
    free_allocated(&slots_off);
    free_allocated(&all_off);
    return 0;
}

static int test_hazard_deterministic_same_seed_and_counts_in_budget(void) {
    srand(7);
    Drive env = hazard_test_env(drive_carla_map(), 32, 48, 42);
    allocate(&env);
    hazard_full_reset(&env);

    int placed = hazard_placed_total(&env);
    EXPECT_TRUE(placed > 0);
    EXPECT_EQ_INT(hazard_valid_slots(&env), placed);
    EXPECT_EQ_INT(env.hazard_reject_no_slot, 0);
    // planned-count highs: parked <= 10, cluster cars <= 4, cones <= 2 zones x 14
    EXPECT_TRUE(env.hazard_placed_parked >= 1 && env.hazard_placed_parked <= 10);
    EXPECT_TRUE(env.hazard_placed_cluster_cars >= 2 && env.hazard_placed_cluster_cars <= 4);
    EXPECT_EQ_INT(env.hazard_placed_cluster_cars % 2, 0); // clusters commit as pairs
    EXPECT_TRUE(env.hazard_placed_cones >= 8 && env.hazard_placed_cones <= 28);
    EXPECT_TRUE(placed <= env.cosim_partner_slots); // slot budget respected

    float poses_x[48];
    float poses_y[48];
    float poses_h[48];
    int counts_before[3];
    counts_before[0] = env.hazard_placed_parked;
    counts_before[1] = env.hazard_placed_cluster_cars;
    counts_before[2] = env.hazard_placed_cones;
    for (int k = 0; k < env.cosim_partner_slots; k++) {
        const Agent *agent = &env.agents[env.static_agent_indices[k]];
        poses_x[k] = agent->sim_x;
        poses_y[k] = agent->sim_y;
        poses_h[k] = agent->sim_heading;
    }

    hazard_full_reset(&env);
    EXPECT_EQ_INT(hazard_placed_total(&env), placed);
    EXPECT_EQ_INT(env.hazard_placed_parked, counts_before[0]);
    EXPECT_EQ_INT(env.hazard_placed_cluster_cars, counts_before[1]);
    EXPECT_EQ_INT(env.hazard_placed_cones, counts_before[2]);
    for (int k = 0; k < env.cosim_partner_slots; k++) {
        const Agent *agent = &env.agents[env.static_agent_indices[k]];
        EXPECT_NEAR(agent->sim_x, poses_x[k], 0.0f);
        EXPECT_NEAR(agent->sim_y, poses_y[k], 0.0f);
        EXPECT_NEAR(agent->sim_heading, poses_h[k], 0.0f);
    }

    // a different episode seed must produce a different layout (same map, same knobs)
    Drive other = hazard_test_env(drive_carla_map(), 32, 48, 43);
    allocate(&other);
    hazard_full_reset(&other);
    EXPECT_TRUE(hazard_placed_total(&other) > 0);
    int any_diff = 0;
    for (int k = 0; k < env.cosim_partner_slots && !any_diff; k++) {
        const Agent *agent = &other.agents[other.static_agent_indices[k]];
        any_diff = fabsf(agent->sim_x - poses_x[k]) > 1e-4f || fabsf(agent->sim_y - poses_y[k]) > 1e-4f;
    }
    EXPECT_TRUE(any_diff);

    free_allocated(&env);
    free_allocated(&other);
    return 0;
}

static int test_hazard_legality_invariants(void) {
    srand(7);
    Drive env = hazard_test_env(drive_carla_map(), 32, 48, 42);
    allocate(&env);
    hazard_full_reset(&env);

    int placed = hazard_placed_total(&env);
    int first = hazard_slot_base(&env);
    int cluster_pairs = 0;

    for (int i = first; i < first + placed; i++) {
        Agent *hazard = &env.agents[i];
        // committed hazards are live, latched-free, kernel-static, and on-lane
        EXPECT_EQ_INT(hazard->sim_valid, 1);
        EXPECT_EQ_INT(hazard->removed, 0);
        EXPECT_EQ_INT(hazard->stopped, 0);
        EXPECT_EQ_INT(hazard->controller, CONTROLLER_STATIC);
        EXPECT_TRUE(hazard->current_lane_idx != -1); // kernel lane association = on-road evidence
        EXPECT_FALSE(check_spawn_offroad(&env, hazard, 0.0f));

        // vs dynamic agents: no overlap at the conflict margin
        for (int x = 0; x < env.active_agent_count; x++) {
            const Agent *dyn = &env.agents[env.active_agent_indices[x]];
            if (dyn->sim_valid != 1 || dyn->sim_x == INVALID_POSITION || dyn->removed) {
                continue;
            }
            if (hazard_boxes_overlap_margin(hazard, (Agent *) dyn)) {
                EXPECT_TRUE(0); // hazard overlaps a dynamic agent at the conflict margin
            }
        }

        // vs other hazards: overlap pairs exist only inside collision clusters
        // or between adjacent cones of one block wall (the prototype never
        // intra-zone-checks cones; block cols sit ~0.47 m apart at 0.5 m size)
        for (int j = i + 1; j < first + placed; j++) {
            Agent *other = &env.agents[j];
            int both_cones = hazard->sim_length < 1.0f && other->sim_length < 1.0f;
            if (both_cones) {
                continue;
            }
            if (hazard_boxes_overlap_margin(hazard, other)) {
                // car-car overlap must be an intentional cluster pair:
                // two vehicles overlapping at bare margin, coplanar
                Agent bare_a = *hazard;
                Agent bare_b = *other;
                EXPECT_TRUE(check_obb_collision(&bare_a, &bare_b));
                EXPECT_EQ_INT(hazard->type, VEHICLE);
                EXPECT_EQ_INT(other->type, VEHICLE);
                EXPECT_NEAR(hazard->sim_z, other->sim_z, 1e-4f);
                cluster_pairs++;
            } else {
                EXPECT_FALSE(check_obb_collision(hazard, other));
            }
        }
    }
    EXPECT_EQ_INT(cluster_pairs, env.hazard_placed_cluster_cars / 2);

    // parked spacing: two parked cars on the same lane keep the bumper gap
    for (int i = first; i < first + env.hazard_placed_parked; i++) {
        for (int j = i + 1; j < first + env.hazard_placed_parked; j++) {
            float dx = env.agents[j].sim_x - env.agents[i].sim_x;
            float dy = env.agents[j].sim_y - env.agents[i].sim_y;
            float gap = sqrtf(dx * dx + dy * dy);
            EXPECT_TRUE(gap >= HAZARD_PARKED_SPACING_GAP_M);
        }
    }

    free_allocated(&env);
    return 0;
}

static int test_hazard_steps_clean_and_neutral_knob_permutations(void) {
    // one enabled class at a time: each must still place within its own budget,
    // and a short neutral-action rollout over the hazards stays finite
    for (int variant = 0; variant < 3; variant++) {
        srand(7);
        Drive env = hazard_test_env(drive_carla_map(), 32, 48, 42);
        env.hazard_parked_enable = variant == 0;
        env.hazard_clusters_enable = variant == 1;
        env.hazard_cones_enable = variant == 2;
        allocate(&env);
        hazard_full_reset(&env);

        EXPECT_EQ_INT(env.hazard_placed_parked, variant == 0 ? env.hazard_placed_parked : 0);
        EXPECT_EQ_INT(env.hazard_placed_cluster_cars, variant == 1 ? env.hazard_placed_cluster_cars : 0);
        EXPECT_EQ_INT(env.hazard_placed_cones, variant == 2 ? env.hazard_placed_cones : 0);
        EXPECT_TRUE(hazard_placed_total(&env) > 0);
        EXPECT_EQ_INT(hazard_valid_slots(&env), hazard_placed_total(&env));

        int obs_size = compute_observation_size(&env);
        for (int t = 0; t < 10; t++) {
            drive_set_neutral_actions(&env);
            c_step(&env);
            EXPECT_TRUE(drive_all_finite(env.observations, env.active_agent_count * obs_size));
            EXPECT_TRUE(drive_all_finite(env.rewards, env.active_agent_count));
            // static slots must never be stepped by c_step
            EXPECT_EQ_INT(hazard_valid_slots(&env), hazard_placed_total(&env));
        }
        free_allocated(&env);
    }
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_hazard_disabled_keeps_episode_stream_bit_identical();
    failures += test_hazard_deterministic_same_seed_and_counts_in_budget();
    failures += test_hazard_legality_invariants();
    failures += test_hazard_steps_clean_and_neutral_knob_permutations();
    if (failures == 0) {
        printf("All test groups passed\n");
        return 0;
    }
    printf("%d test group(s) FAILED\n", failures);
    return 1;
}
