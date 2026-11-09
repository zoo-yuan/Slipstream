#include "include/drive_fixture.h"
#include "include/test.h"

// Stop line spans y in [-2, 2] at x = 0, traffic flows +x; the trigger box is centred on it,
// 1.7 m deep on each side. Agents are 4 m long, so a stop with the nose at the line overlaps it.
static int stop_lanes[1] = {100};

static TrafficControlElement drive_test_stop_sign(void) {
    TrafficControlElement tc = {0};
    tc.type = TRAFFIC_CONTROL_TYPE_STOP_SIGN;
    tc.stop_line[1] = -2.0f;
    tc.stop_line[4] = 2.0f;
    tc.heading = 0.0f;
    tc.num_controlled_lanes = 1;
    tc.controlled_lanes = stop_lanes;
    return tc;
}

static Drive drive_test_stop_env(TrafficControlElement *tc, Agent *agent) {
    Drive env = {0};
    env.num_traffic_elements = 1;
    env.traffic_elements = tc;
    env.agents = agent;
    return env;
}

static Agent drive_test_stop_agent(float x, float y, float heading) {
    Agent agent = drive_test_agent(x, y, heading);
    reset_agent_state(&agent);
    agent.current_lane_idx = 100;
    agent.previous_lane_idx = 100;
    return agent;
}

// One sim step: the agent moved prev -> cur with the given speed (positive = forward).
static void drive_test_step_to(Agent *agent, float x, float y, float speed) {
    copy_pose_to_prev(agent);
    agent->sim_x = x;
    agent->sim_y = y;
    agent->sim_speed = fabsf(speed);
    agent->sim_speed_signed = speed;
}

static void drive_test_turn_to(Agent *agent, float heading) {
    agent->sim_heading = heading;
    agent->cos_heading = cosf(heading);
    agent->sin_heading = sinf(heading);
}

static int test_approach_acquires_target(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-15.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -14.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, 0);
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0);
    return 0;
}

static int test_far_sign_not_targeted(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-26.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -25.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_stop_in_box_then_cross_no_flag(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    // nose 0.5 m before the line, standing still
    drive_test_step_to(&agent, -2.5f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 1);
    drive_test_step_to(&agent, -0.5f, 0.0f, 3.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 0.0f, 3.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, 0);
    EXPECT_EQ_INT(agent.stop_sign_last_failed_idx, -1);
    return 0;
}

static int test_cross_without_stop_flags_once(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, -0.5f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 0.0f, 10.0f);
    EXPECT_TRUE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    EXPECT_EQ_INT(agent.stop_sign_last_failed_idx, 0);
    // still inside the scaled box: the run sign is not re-targeted, nothing fires twice
    drive_test_step_to(&agent, 1.5f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_stop_before_box_does_not_count(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    // nose 4 m before the line: outside the 1.7 m deep trigger box
    drive_test_step_to(&agent, -6.0f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0);
    drive_test_step_to(&agent, -0.5f, 0.0f, 5.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 0.0f, 5.0f);
    EXPECT_TRUE(update_stop_sign_state(&env, 0));
    return 0;
}

static int test_slow_roll_in_box_does_not_count(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, -2.5f, 0.0f, 0.3f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0);
    drive_test_step_to(&agent, -0.5f, 0.0f, 0.3f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 0.0f, 0.3f);
    EXPECT_TRUE(update_stop_sign_state(&env, 0));
    return 0;
}

static int test_swerve_around_line_flags(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    // acquired on the lane, then crossing 6 m beside the painted line
    drive_test_step_to(&agent, -0.5f, 6.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 6.0f, 10.0f);
    EXPECT_TRUE(update_stop_sign_state(&env, 0));
    return 0;
}

static int test_parallel_lane_never_targeted(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 6.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 6.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    drive_test_step_to(&agent, -0.5f, 6.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 6.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    return 0;
}

static int test_opposing_direction_not_targeted(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(6.0f, 0.0f, (float) M_PI);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, 5.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_cross_street_not_targeted(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(1.0f, -6.0f, (float) M_PI / 2.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, 1.0f, -5.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_reversing_not_targeted(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-4.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -5.0f, 0.0f, -1.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_different_z_level_not_targeted(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    agent.sim_z = 6.0f;
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_state_clears_beyond_proximity(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, -0.5f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 0.0f, 10.0f);
    EXPECT_TRUE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 25.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_last_failed_idx, -1);
    // a completed stop is forgotten the same way, so a later approach must stop again
    Agent again = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    env.agents = &again;
    drive_test_step_to(&again, -2.5f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(again.stop_sign_stop_completed, 1);
    drive_test_step_to(&again, 25.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(again.stop_sign_target_idx, -1);
    EXPECT_EQ_INT(again.stop_sign_stop_completed, 0);
    return 0;
}

static int test_spawn_inside_box_counts_as_stopped(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-1.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 1);
    drive_test_step_to(&agent, 1.0f, 0.0f, 2.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    return 0;
}

// Mid U-turn: standing still inside the box while pointed 115 deg off the lane, then aligning and crossing.
static int test_standstill_before_targeting_counts(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-0.7f, 0.0f, 2.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -0.7f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    EXPECT_EQ_INT(agent.stop_sign_standstill_idx, 0);
    drive_test_turn_to(&agent, 0.0f);
    drive_test_step_to(&agent, -0.3f, 0.0f, 1.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, 0);
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 1);
    drive_test_step_to(&agent, 0.5f, 0.0f, 1.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_last_failed_idx, -1);
    return 0;
}

static int test_standstill_outside_box_before_targeting_does_not_count(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-6.0f, 0.0f, 2.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -6.0f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_standstill_idx, -1);
    drive_test_turn_to(&agent, 0.0f);
    drive_test_step_to(&agent, -0.5f, 0.0f, 5.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0);
    drive_test_step_to(&agent, 0.5f, 0.0f, 5.0f);
    EXPECT_TRUE(update_stop_sign_state(&env, 0));
    return 0;
}

static int test_standstill_clears_beyond_proximity(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-0.7f, 0.0f, 2.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -0.7f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_standstill_idx, 0);
    drive_test_step_to(&agent, -25.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_standstill_idx, -1);
    return 0;
}

// Two lanes' signs 4 m apart: 19 m out on a curved approach the agent sits in the neighbour line's frame
// (|across| 0.1 m) and latches it; once it is in its own lane the own sign must take over and complete.
static int test_neighbour_lane_target_yields_to_own_lane_sign(void) {
    TrafficControlElement tcs[2];
    tcs[0] = drive_test_stop_sign();
    tcs[0].stop_line[1] = 2.0f;
    tcs[0].stop_line[4] = 6.0f;
    tcs[1] = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-20.0f, 3.9f, 0.0f);
    Drive env = drive_test_stop_env(tcs, &agent);
    env.num_traffic_elements = 2;
    drive_test_step_to(&agent, -19.0f, 3.9f, 8.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, 0);
    drive_test_step_to(&agent, -10.0f, 0.5f, 8.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, 1);
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0);
    drive_test_step_to(&agent, -2.5f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, 1);
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 1);
    drive_test_step_to(&agent, 0.5f, 0.0f, 3.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_last_failed_idx, -1);
    return 0;
}

static int test_traffic_light_ignored(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    tc.type = TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_obb_collision_unchanged_by_shared_sat(void) {
    Agent ego = drive_test_agent(0.0f, 0.0f, 0.0f);
    Agent other = drive_test_agent(3.0f, 1.5f, 0.5f);
    EXPECT_TRUE(check_obb_collision(&ego, &other));
    other = drive_test_agent(6.0f, 0.0f, 0.0f);
    EXPECT_FALSE(check_obb_collision(&ego, &other));
    return 0;
}

// --- K3 sampled dwell: per-sign hold-below-speed requirement ---

static Drive drive_test_dwell_env(TrafficControlElement *tc, Agent *agent, float dwell_seconds) {
    Drive env = drive_test_stop_env(tc, agent);
    env.dt = 0.1f;
    tc->stop_dwell_seconds = dwell_seconds;
    return env;
}

// dwell 0 (legacy default): the first below-speed tick inside the box completes the stop.
static int test_dwell_zero_completes_on_first_standstill_tick(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_dwell_env(&tc, &agent, 0.0f);
    drive_test_step_to(&agent, -2.5f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 1);
    EXPECT_NEAR(agent.stop_sign_standstill_dwell_seconds, 0.1f, 1e-6f);
    return 0;
}

// dwell 0.4 s: the hold must accumulate four qualifying ticks before completion; crossing early
// still runs the sign.
static int test_dwell_hold_duration_required(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_dwell_env(&tc, &agent, 0.4f);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    for (int tick = 1; tick <= 3; tick++) {
        drive_test_step_to(&agent, -2.5f, 0.0f, 0.0f);
        EXPECT_FALSE(update_stop_sign_state(&env, 0));
        EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0);
    }
    drive_test_step_to(&agent, -2.5f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 1);
    EXPECT_NEAR(agent.stop_sign_standstill_dwell_seconds, 0.4f, 1e-5f);
    // completed stop -> crossing is clean
    drive_test_step_to(&agent, 0.5f, 0.0f, 3.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_last_failed_idx, -1);
    return 0;
}

// A partial hold survives a short excursion (latch persists within the 20 m proximity) and
// resumes accumulation on return.
static int test_dwell_partial_hold_persists_within_proximity(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_dwell_env(&tc, &agent, 0.4f);
    // two hold ticks (0.2 s accumulated)
    drive_test_step_to(&agent, -2.5f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, -2.5f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    // creep backward out of the box (above stop speed, still within the 20 m proximity; a forward
    // creep would run the uncompleted sign, which is a different, already-covered violation)
    drive_test_step_to(&agent, -6.0f, 0.0f, -5.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_standstill_idx, 0);
    EXPECT_NEAR(agent.stop_sign_standstill_dwell_seconds, 0.2f, 1e-5f);
    // return and hold two more ticks -> 0.4 s -> completed
    drive_test_step_to(&agent, -2.5f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0);
    drive_test_step_to(&agent, -2.5f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 1);
    return 0;
}

// The hold latch is keyed to one box at a time: shifting into another sign's box restarts the
// accumulation, so time held at sign A never credits sign B.
static int test_dwell_restarts_on_other_sign_box(void) {
    static int lanes_a[1] = {100};
    static int lanes_b[1] = {101};
    TrafficControlElement signs[2];
    signs[0] = drive_test_stop_sign();  // sign A: line y in [-2, 2] at x = 0 (own lane)
    signs[1] = drive_test_stop_sign();  // sign B: parallel lane, line y in [1.5, 5.5]
    signs[0].controlled_lanes = lanes_a;
    signs[1].controlled_lanes = lanes_b;
    signs[1].stop_line[1] = 1.5f;
    signs[1].stop_line[4] = 5.5f;
    signs[0].stop_dwell_seconds = 0.4f;
    signs[1].stop_dwell_seconds = 0.4f;
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(signs, &agent);
    // The fixture defaults to one element; sign B must be visible to the standstill/target scans
    // for the latch to move from A to B (kernel scans all num_traffic_elements).
    env.num_traffic_elements = 2;
    env.dt = 0.1f;
    // hold two ticks in sign A's box (0.2 s accumulated against the A latch)
    drive_test_step_to(&agent, -2.5f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_standstill_idx, 0);
    drive_test_step_to(&agent, -2.5f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_NEAR(agent.stop_sign_standstill_dwell_seconds, 0.2f, 1e-5f);
    // shift laterally into sign B's box: the latch moves and the accumulation restarts, so the
    // A hold must not complete B; B is the new target (A is now 3.5 m off-axis) needing 0.4 s
    drive_test_step_to(&agent, -2.5f, 3.5f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_standstill_idx, 1);
    EXPECT_EQ_INT(agent.stop_sign_target_idx, 1);
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0);
    EXPECT_NEAR(agent.stop_sign_standstill_dwell_seconds, 0.1f, 1e-5f);
    for (int tick = 0; tick < 3; tick++) {
        drive_test_step_to(&agent, -2.5f, 3.5f, 0.0f);
        EXPECT_FALSE(update_stop_sign_state(&env, 0));
    }
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 1);
    return 0;
}

// Standstill memory credit becomes dwell-aware: a pre-acquisition hold shorter than the sign's
// dwell acquires the target without completing; the hold then finishes under the target.
static int test_dwell_standstill_memory_credit_needs_full_hold(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-0.7f, 0.0f, 2.0f); // misaligned: no target while holding
    Drive env = drive_test_dwell_env(&tc, &agent, 0.4f);
    drive_test_step_to(&agent, -0.7f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_standstill_idx, 0);
    drive_test_step_to(&agent, -0.7f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_NEAR(agent.stop_sign_standstill_dwell_seconds, 0.2f, 1e-5f);
    drive_test_turn_to(&agent, 0.0f);
    drive_test_step_to(&agent, -0.3f, 0.0f, 1.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, 0);
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0); // 0.2 s < 0.4 s: no instant credit
    drive_test_step_to(&agent, -0.3f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, -0.3f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 1);
    return 0;
}

// Reset-side sampling: active range draws a per-sign value inside [lo, hi] deterministically;
// the legacy range [0,0] draws nothing (RNG stream untouched, dwells stay 0).
static int test_dwell_sampling_range_and_rng_gate(void) {
    static int sign_lanes[3] = {100, 101, 102};
    TrafficControlElement signs[3];
    for (int i = 0; i < 3; i++) {
        signs[i] = drive_test_stop_sign();
        signs[i].controlled_lanes = &sign_lanes[i];
    }
    Drive env = {0};
    env.num_traffic_elements = 3;
    env.traffic_elements = signs;
    env.dt = 0.1f;

    rng_seed(&env.rng_state, 11);
    Rng before = env.rng_state;
    sample_stop_sign_dwells(&env); // legacy [0,0]: no draws
    EXPECT_EQ_INT(memcmp(&env.rng_state, &before, sizeof(Rng)), 0);
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(signs[i].stop_dwell_seconds, 0.0f, 0.0f);
    }

    env.stop_sign_dwell_range_lo = 0.2f;
    env.stop_sign_dwell_range_hi = 0.6f;
    rng_seed(&env.rng_state, 11);
    sample_stop_sign_dwells(&env);
    Rng after_draw = env.rng_state;
    float drawn[3];
    for (int i = 0; i < 3; i++) {
        EXPECT_TRUE(signs[i].stop_dwell_seconds >= 0.2f && signs[i].stop_dwell_seconds <= 0.6f);
        drawn[i] = signs[i].stop_dwell_seconds;
    }
    EXPECT_EQ_INT(memcmp(&env.rng_state, &before, sizeof(Rng)) != 0, 1);
    // deterministic under the same seed, and the three signs are drawn from one stream
    rng_seed(&env.rng_state, 11);
    sample_stop_sign_dwells(&env);
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(signs[i].stop_dwell_seconds, drawn[i], 0.0f);
    }
    EXPECT_EQ_INT(memcmp(&env.rng_state, &after_draw, sizeof(Rng)), 0);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_approach_acquires_target);
    RUN_TEST(test_far_sign_not_targeted);
    RUN_TEST(test_stop_in_box_then_cross_no_flag);
    RUN_TEST(test_cross_without_stop_flags_once);
    RUN_TEST(test_stop_before_box_does_not_count);
    RUN_TEST(test_slow_roll_in_box_does_not_count);
    RUN_TEST(test_swerve_around_line_flags);
    RUN_TEST(test_parallel_lane_never_targeted);
    RUN_TEST(test_opposing_direction_not_targeted);
    RUN_TEST(test_cross_street_not_targeted);
    RUN_TEST(test_reversing_not_targeted);
    RUN_TEST(test_different_z_level_not_targeted);
    RUN_TEST(test_state_clears_beyond_proximity);
    RUN_TEST(test_spawn_inside_box_counts_as_stopped);
    RUN_TEST(test_standstill_before_targeting_counts);
    RUN_TEST(test_neighbour_lane_target_yields_to_own_lane_sign);
    RUN_TEST(test_standstill_outside_box_before_targeting_does_not_count);
    RUN_TEST(test_standstill_clears_beyond_proximity);
    RUN_TEST(test_traffic_light_ignored);
    RUN_TEST(test_obb_collision_unchanged_by_shared_sat);
    RUN_TEST(test_dwell_zero_completes_on_first_standstill_tick);
    RUN_TEST(test_dwell_hold_duration_required);
    RUN_TEST(test_dwell_partial_hold_persists_within_proximity);
    RUN_TEST(test_dwell_restarts_on_other_sign_box);
    RUN_TEST(test_dwell_standstill_memory_credit_needs_full_hold);
    RUN_TEST(test_dwell_sampling_range_and_rng_gate);
    return test_summary(failures);
}
