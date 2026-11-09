#include "include/drive_fixture.h"
#include "include/test.h"

// K3 pedestrian exemption gate + all-knobs-off stream bit-identity.
//
// TerraZero per-class rule: pedestrians are exempt from the intersection right-of-way
// infractions (road exit / red light / stop sign); cyclists stay enforced as vehicles and
// collisions apply to everyone. The gate is inert in today's flows (gigaflow spawns
// VEHICLE-typed actives only; K2 jaywalkers are static slots that never reach
// compute_metrics) and paves the way for sim-agent active pedestrians.
//
// The stream test pins the iron rule: all four K3 knobs at their defaults (truth timing off,
// footprint sweep off, dwell range [0,0], ped exemption ON) leave the episode stream
// bit-identical to a knob-free env - the ped gate consumes no RNG draws and no active
// PEDESTRIAN exists for it to exempt.

// --- shared helpers -----------------------------------------------------------------

// Same map -> same element layout: copy only the signal geometry fields. A struct copy would
// alias the source env's states/controlled_lanes pointers and corrupt free_allocated, so every
// env keeps ownership of its own heap states buffer (filled in place by the callers).
static void ped_copy_signal_geometry(TrafficControlElement *dst, const TrafficControlElement *src) {
    dst->type = src->type;
    for (int k = 0; k < 6; k++) {
        dst->stop_line[k] = src->stop_line[k];
    }
    dst->heading = src->heading;
}

static void ped_fill_red(TrafficControlElement *tc) {
    for (int t = 0; t < tc->state_size; t++) {
        tc->states[t] = TRAFFIC_CONTROL_STATE_RED;
    }
}

static Drive ped_test_env(int pedestrian_exempt) {
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.pedestrian_traffic_exempt = pedestrian_exempt;
    return env;
}

static int ped_first_lane_element(const Drive *env) {
    for (int i = 0; i < env->num_road_elements; i++) {
        if (is_road_lane(env->road_elements[i].type)) {
            return i;
        }
    }
    return -1;
}

// Place the env's single active agent on a lane point with a prev -> cur motion step.
static void ped_place_agent(Drive *env, int type, float x, float y, float z, float heading, float step_x) {
    Agent *agent = &env->agents[env->active_agent_indices[0]];
    agent->type = type;
    agent->sim_x = x;
    agent->sim_y = y;
    agent->sim_z = z;
    agent->sim_heading = heading;
    agent->cos_heading = cosf(heading);
    agent->sim_speed = 0.0f;
    agent->sim_speed_signed = 0.0f;
    agent->sin_heading = sinf(heading);
    copy_pose_to_prev(agent);
    agent->sim_x = x + step_x; // the step the enforcement checks see
}

// --- road-exit (offgrid) entry ------------------------------------------------------

static int test_ped_exempt_offgrid_offroad_not_flagged(void) {
    Drive env = ped_test_env(1);
    int lane_idx = ped_first_lane_element(&env);
    EXPECT_TRUE(lane_idx >= 0);
    ped_place_agent(&env, PEDESTRIAN, 1.0e4f, 1.0e4f, 0.0f, 0.0f, 0.0f);
    compute_metrics(&env, env.active_agent_indices[0], 0);
    const Agent *agent = &env.agents[env.active_agent_indices[0]];
    EXPECT_NEAR(agent->metrics_array[OFFROAD_IDX], 0.0f, 0.0f);
    EXPECT_EQ_INT(agent->stopped || agent->removed, 0);

    // gate off restores the pre-K3 behaviour: the pedestrian is flagged like a vehicle
    Drive gated = ped_test_env(0);
    ped_place_agent(&gated, PEDESTRIAN, 1.0e4f, 1.0e4f, 0.0f, 0.0f, 0.0f);
    compute_metrics(&gated, gated.active_agent_indices[0], 0);
    EXPECT_NEAR(gated.agents[gated.active_agent_indices[0]].metrics_array[OFFROAD_IDX], 1.0f, 0.0f);
    free_allocated(&env);
    free_allocated(&gated);
    return 0;
}

// --- red-light entry ----------------------------------------------------------------

static int test_ped_exempt_red_light_not_flagged(void) {
    Drive env = ped_test_env(1);
    int lane_idx = ped_first_lane_element(&env);
    EXPECT_TRUE(lane_idx >= 0);
    const RoadMapElement *lane = &env.road_elements[lane_idx];
    float px = lane->x[0], py = lane->y[0], pz = lane->z[0], heading = lane->headings[0];

    // turn element 0 (a CARLA traffic light; the geometry-only bin leaves states to the gigaflow
    // fill) into an all-red program, written into the env's own heap states buffer
    TrafficControlElement *tc = &env.traffic_elements[0];
    tc->type = TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    float bar_x = -sinf(heading), bar_y = cosf(heading);
    tc->stop_line[0] = px - 2.0f * bar_x;
    tc->stop_line[1] = py - 2.0f * bar_y;
    tc->stop_line[2] = pz;
    tc->stop_line[3] = px + 2.0f * bar_x;
    tc->stop_line[4] = py + 2.0f * bar_y;
    tc->stop_line[5] = pz;
    tc->heading = heading;
    ped_fill_red(tc);
    env.timestep = 5;

    ped_place_agent(&env, PEDESTRIAN, px - 0.5f, py, pz, heading, 1.0f);
    compute_metrics(&env, env.active_agent_indices[0], 0);
    EXPECT_NEAR(env.agents[env.active_agent_indices[0]].metrics_array[RED_LIGHT_IDX], 0.0f, 0.0f);

    // gate off: the same motion on red flags
    Drive gated = ped_test_env(0);
    ped_copy_signal_geometry(&gated.traffic_elements[0], tc);
    ped_fill_red(&gated.traffic_elements[0]);
    ped_place_agent(&gated, PEDESTRIAN, px - 0.5f, py, pz, heading, 1.0f);
    gated.timestep = 5;
    compute_metrics(&gated, gated.active_agent_indices[0], 0);
    EXPECT_NEAR(gated.agents[gated.active_agent_indices[0]].metrics_array[RED_LIGHT_IDX], 1.0f, 0.0f);

    // cyclists are NOT exempt even with the gate on
    Drive cyclist = ped_test_env(1);
    ped_copy_signal_geometry(&cyclist.traffic_elements[0], tc);
    ped_fill_red(&cyclist.traffic_elements[0]);
    ped_place_agent(&cyclist, CYCLIST, px - 0.5f, py, pz, heading, 1.0f);
    cyclist.timestep = 5;
    compute_metrics(&cyclist, cyclist.active_agent_indices[0], 0);
    EXPECT_NEAR(cyclist.agents[cyclist.active_agent_indices[0]].metrics_array[RED_LIGHT_IDX], 1.0f, 0.0f);

    // vehicles are NOT exempt even with the gate on
    Drive vehicle = ped_test_env(1);
    ped_copy_signal_geometry(&vehicle.traffic_elements[0], tc);
    ped_fill_red(&vehicle.traffic_elements[0]);
    ped_place_agent(&vehicle, VEHICLE, px - 0.5f, py, pz, heading, 1.0f);
    vehicle.timestep = 5;
    compute_metrics(&vehicle, vehicle.active_agent_indices[0], 0);
    EXPECT_NEAR(vehicle.agents[vehicle.active_agent_indices[0]].metrics_array[RED_LIGHT_IDX], 1.0f, 0.0f);

    free_allocated(&env);
    free_allocated(&gated);
    free_allocated(&cyclist);
    free_allocated(&vehicle);
    return 0;
}

// --- stop-sign entry ----------------------------------------------------------------

static int test_ped_exempt_stop_sign_not_flagged(void) {
    // the sign logic never reads states, so element 0's heap buffer is left untouched
    Drive env = ped_test_env(1);
    env.stop_signs_enabled = 1;
    int lane_idx = ped_first_lane_element(&env);
    EXPECT_TRUE(lane_idx >= 0);
    const RoadMapElement *lane = &env.road_elements[lane_idx];
    float px = lane->x[0], py = lane->y[0], pz = lane->z[0], heading = lane->headings[0];

    TrafficControlElement *tc = &env.traffic_elements[0];
    tc->type = TRAFFIC_CONTROL_TYPE_STOP_SIGN;
    float bar_x = -sinf(heading), bar_y = cosf(heading);
    tc->stop_line[0] = px - 2.0f * bar_x;
    tc->stop_line[1] = py - 2.0f * bar_y;
    tc->stop_line[2] = pz;
    tc->stop_line[3] = px + 2.0f * bar_x;
    tc->stop_line[4] = py + 2.0f * bar_y;
    tc->stop_line[5] = pz;
    tc->heading = heading;
    env.timestep = 0;

    // a below-speed pedestrian crossing the sign's line without a completed stop
    ped_place_agent(&env, PEDESTRIAN, px - 0.5f, py, pz, heading, 1.0f);
    env.agents[env.active_agent_indices[0]].sim_speed = 3.0f;
    env.agents[env.active_agent_indices[0]].sim_speed_signed = 3.0f;
    compute_metrics(&env, env.active_agent_indices[0], 0);
    EXPECT_NEAR(env.agents[env.active_agent_indices[0]].metrics_array[STOP_SIGN_IDX], 0.0f, 0.0f);

    // gate off: the same run flags
    Drive gated = ped_test_env(0);
    gated.stop_signs_enabled = 1;
    ped_copy_signal_geometry(&gated.traffic_elements[0], tc);
    ped_place_agent(&gated, PEDESTRIAN, px - 0.5f, py, pz, heading, 1.0f);
    gated.agents[gated.active_agent_indices[0]].sim_speed = 3.0f;
    gated.agents[gated.active_agent_indices[0]].sim_speed_signed = 3.0f;
    gated.timestep = 0;
    compute_metrics(&gated, gated.active_agent_indices[0], 0);
    EXPECT_NEAR(gated.agents[gated.active_agent_indices[0]].metrics_array[STOP_SIGN_IDX], 1.0f, 0.0f);

    // cyclists stay enforced: the same run flags with the gate on
    Drive cyclist = ped_test_env(1);
    cyclist.stop_signs_enabled = 1;
    ped_copy_signal_geometry(&cyclist.traffic_elements[0], tc);
    ped_place_agent(&cyclist, CYCLIST, px - 0.5f, py, pz, heading, 1.0f);
    cyclist.agents[cyclist.active_agent_indices[0]].sim_speed = 3.0f;
    cyclist.agents[cyclist.active_agent_indices[0]].sim_speed_signed = 3.0f;
    cyclist.timestep = 0;
    compute_metrics(&cyclist, cyclist.active_agent_indices[0], 0);
    EXPECT_NEAR(cyclist.agents[cyclist.active_agent_indices[0]].metrics_array[STOP_SIGN_IDX], 1.0f, 0.0f);

    free_allocated(&env);
    free_allocated(&gated);
    free_allocated(&cyclist);
    return 0;
}

// --- K3 iron rule: all knobs at defaults leave the stream bit-identical ---------------

static int k3_streams_bit_identical(const Drive *a, const Drive *b) {
    int obs_size = compute_observation_size((Drive *) a);
    size_t bytes = (size_t) a->active_agent_count * (size_t) obs_size * sizeof(float);
    return memcmp(a->observations, b->observations, bytes) == 0;
}

static int test_k3_defaults_keep_episode_stream_bit_identical(void) {
    srand(7);
    Drive bare = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 16, 0);
    bare.init_seed = 42;
    bare.use_exact_episode_seed = 1;

    // same config with every K3 knob pinned to its documented default
    Drive pinned = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 16, 0);
    pinned.init_seed = 42;
    pinned.use_exact_episode_seed = 1;
    pinned.traffic_light_truth_timing = 0;
    pinned.traffic_light_footprint_sweep = 0;
    pinned.stop_sign_dwell_range_lo = 0.0f;
    pinned.stop_sign_dwell_range_hi = 0.0f;
    pinned.pedestrian_traffic_exempt = 1; // default ON: inert without active pedestrians

    allocate(&bare);
    allocate(&pinned);
    bare.timestep = -1; // full-path reset (mirrors binding my_reseed)
    pinned.timestep = -1;
    c_reset(&bare);
    c_reset(&pinned);

    EXPECT_EQ_INT(pinned.truth_timing_fallback_lights, 0);
    EXPECT_TRUE(k3_streams_bit_identical(&bare, &pinned));
    for (int x = 0; x < bare.active_agent_count; x++) {
        const Agent *a = &bare.agents[bare.active_agent_indices[x]];
        const Agent *b = &pinned.agents[pinned.active_agent_indices[x]];
        EXPECT_NEAR(a->sim_x, b->sim_x, 0.0f);
        EXPECT_NEAR(a->sim_y, b->sim_y, 0.0f);
        EXPECT_NEAR(a->sim_heading, b->sim_heading, 0.0f);
    }
    int obs_size = compute_observation_size(&bare);
    for (int t = 0; t < 12; t++) {
        drive_set_neutral_actions(&bare);
        drive_set_neutral_actions(&pinned);
        c_step(&bare);
        c_step(&pinned);
        EXPECT_TRUE(k3_streams_bit_identical(&bare, &pinned));
        EXPECT_TRUE(drive_all_finite(pinned.observations, pinned.active_agent_count * obs_size));
    }
    free_allocated(&bare);
    free_allocated(&pinned);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_ped_exempt_offgrid_offroad_not_flagged);
    RUN_TEST(test_ped_exempt_red_light_not_flagged);
    RUN_TEST(test_ped_exempt_stop_sign_not_flagged);
    RUN_TEST(test_k3_defaults_keep_episode_stream_bit_identical);
    return test_summary(failures);
}
