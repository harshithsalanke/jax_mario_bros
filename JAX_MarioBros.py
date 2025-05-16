import sys
import pygame
import chex
import jax
import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass

# --- Bildschirmgröße ---
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 210
WINDOW_SCALE = 3  # Optional zum Hochskalieren

# --- Physik-Parameter ---
MOVE_SPEED = 1
ASCEND_VY = -2.0  # ↑ 2 px / frame
DESCEND_VY = 2.0  # ↓ 2 px / frame
ASCEND_FRAMES = 21  # 42 px tall jump (21 × 2)

# --- Spieler-und-Enemy-Größe ---
PLAYER_SIZE = (9, 21)  # w, h
PLAYER_COLOR = (181, 83, 40)
ENEMY_SIZE = (8, 8)  # w, h

PLATFORMS = jnp.array([
    [0, 168, 160, 24],  # Boden
    [0, 57, 64, 3],  # Plattform 1
    [96, 57, 68, 3],  # Plattform 2
    [31, 95, 97, 3],  # Plattform 3 (hier könnte der Pow Block sein)
    [0, 95, 16, 3],  # Plattform 4
    [144, 95, 18, 3],  # Plattform 5
    [0, 135, 48, 3],  # Plattform 6
    [112, 135, 48, 3]  # Plattform 7
])

# --- Pow_Block ---
POW_BLOCK = jnp.array([[72, 135, 16, 7]])  # x, y, w, h
ENEMY_SPAWN_FRAMES = jnp.array([0, 200])  # example delays for each enemy in frames


# --- GameState mit chex ---
@chex.dataclass
class GameState:
    pos: jnp.ndarray  # [x, y]
    vel: jnp.ndarray  # [vx, vy]
    on_ground: bool
    jump_phase: int
    ascend_frames: int
    enemy_pos: jnp.ndarray  # shape (N, 2)
    enemy_vel: jnp.ndarray  # shape (N, 2)
    enemy_platform_idx: jnp.ndarray  # shape (N,), index into PLATFORMS
    enemy_timer: jnp.ndarray  # shape (N,), counts frames until platform change
    enemy_initial_sides: jnp.ndarray  # NEW: store enemies' initial side (left=0, right=1)
    enemy_delay_timer: jnp.ndarray  # shape (N,), counts frames until enemy starts moving
    pow_hits: int  # or jnp.int32(0)

# --- Initialzustand ---
def init_state():
    platform_1_y = PLATFORMS[1][1]
    platform_2_y = PLATFORMS[2][1]
    pow_hits = 0

    enemy_pos = jnp.array([
        [5.0, platform_1_y - ENEMY_SIZE[1]],
        [130.0, platform_2_y - ENEMY_SIZE[1]]
    ])
    enemy_vel = jnp.array([
        [0.5, 0.0],
        [-0.5, 0.0]
    ])
    enemy_platform_idx = jnp.array([1, 2])
    enemy_timer = jnp.array([0, 0])
    enemy_initial_sides = jnp.array([0, 1])  # NEW: 0=left, 1=right
    enemy_delay_timer = jnp.array([300, 0])  # Delay first enemy 300 frames, second starts immediately

    return GameState(
        # pos=jnp.array([37.0, 74.0]),
        pos=jnp.array([PLATFORMS[0][0], PLATFORMS[0][1] - PLAYER_SIZE[1]]),
        vel=jnp.array([0.0, 0.0]),
        on_ground=False,
        jump_phase=jnp.int32(0),
        ascend_frames=jnp.int32(0),
        enemy_pos=enemy_pos,
        enemy_vel=enemy_vel,
        enemy_platform_idx=enemy_platform_idx,
        enemy_timer=enemy_timer,
        enemy_initial_sides=enemy_initial_sides,
        enemy_delay_timer=enemy_delay_timer,
        pow_hits=jnp.int32(0)
    )

def reset():
    return init_state()

# --- AABB-Kollision: Boden (landed) und Decke (bumped) ---
def check_collision(pos: jnp.ndarray, vel: jnp.ndarray, platforms: jnp.ndarray, pow_block: jnp.ndarray):
    x, y = pos
    vx, vy = vel
    w, h = PLAYER_SIZE

    left, right = x, x + w
    top, bottom = y, y + h

    # Plattformen
    px, py, pw, ph = platforms[:, 0], platforms[:, 1], platforms[:, 2], platforms[:, 3]
    p_left, p_right = px, px + pw
    p_top, p_bottom = py, py + ph

    overlap_x = (right > p_left) & (left < p_right)
    overlap_y = (bottom > p_top) & (top < p_bottom)
    collided = overlap_x & overlap_y

    landed = collided & (vy > 0) & (bottom - vy <= p_top)
    bumped = collided & (vy < 0) & (top - vy >= p_bottom)

    # Höhenkorrektur
    landing_y = jnp.where(landed, p_top - h, jnp.inf)
    bumping_y = jnp.where(bumped, p_bottom, -jnp.inf)
    new_y_land = jnp.min(landing_y)
    new_y_bump = jnp.max(bumping_y)

    # POW block (only bump from below)
    pow_x, pow_y, pow_w, pow_h = pow_block[0]
    pow_left, pow_right = pow_x, pow_x + pow_w
    pow_top, pow_bottom = pow_y, pow_y + pow_h

    pow_overlap_x = (right > pow_left) & (left < pow_right)
    pow_hit_from_below = pow_overlap_x & (vy < 0) & (top - vy >= pow_bottom) & (top <= pow_bottom)
    pow_bump_y = jnp.where(pow_hit_from_below, pow_bottom, -jnp.inf)

    pow_bumped = pow_hit_from_below
    pow_y_new = jnp.max(pow_bump_y)

    return jnp.any(landed), jnp.any(bumped | pow_bumped), new_y_land, jnp.maximum(new_y_bump, pow_y_new), pow_bumped


def check_enemy_collision(player_pos, enemy_pos):
    px, py = player_pos
    pw, ph = PLAYER_SIZE
    ex, ey = enemy_pos[:, 0], enemy_pos[:, 1]
    ew, eh = ENEMY_SIZE

    overlap_x = (px < ex + ew) & (px + pw > ex)
    overlap_y = (py < ey + eh) & (py + ph > ey)
    return jnp.any(overlap_x & overlap_y)


# --- JIT-kompilierte Schritt-Funktion ---
@jit
def step(state: GameState, action: jnp.ndarray) -> GameState:
    move, jump_btn = action[0], action[1].astype(jnp.int32)
    vx = MOVE_SPEED * move
    # -------- phase / frame bookkeeping --------------------------
    start_jump = (jump_btn == 1) & state.on_ground & (state.jump_phase == 0)

    jump_phase = jnp.where(start_jump, 1, state.jump_phase)
    asc_left = jnp.where(start_jump, ASCEND_FRAMES, state.ascend_frames)

    # vertical speed for this frame
    vy = jnp.where(
        jump_phase == 1, ASCEND_VY,
        jnp.where(jump_phase == 2, DESCEND_VY,
                  jnp.where(state.on_ground, 0.0, DESCEND_VY))
    )

    # integrate position
    new_pos = state.pos + jnp.array([vx, vy])

    landed, bumped, y_land, y_bump, pow_hit = check_collision(new_pos, jnp.array([vx, vy]), PLATFORMS, POW_BLOCK)

    new_y = jnp.where(landed, y_land,
                      jnp.where(bumped, y_bump, new_pos[1]))

    # ---------- update phases after collision & time -------------
    asc_left = jnp.where(jump_phase == 1, jnp.maximum(asc_left - 1, 0), asc_left)
    jump_phase = jnp.where((jump_phase == 1) & (asc_left == 0), 2, jump_phase)
    jump_phase = jnp.where(bumped & (vy < 0), 2, jump_phase)
    asc_left = jnp.where(bumped & (vy < 0), 0, asc_left)
    jump_phase = jnp.where(landed, 0, jump_phase)
    asc_left = jnp.where(landed, 0, asc_left)
    jump_phase = jnp.where((jump_phase == 0) & (~landed), 2, jump_phase)

    vy_final = jnp.where(
        jump_phase == 1, ASCEND_VY,
        jnp.where(jump_phase == 2, DESCEND_VY, 0.0)
    )


    # new_x = jnp.clip(new_pos[0], 0, SCREEN_WIDTH - PLAYER_SIZE[0])
    # Wrap around horizontally
    new_x = new_pos[0] % (SCREEN_WIDTH - PLAYER_SIZE[0])
    active_mask = state.enemy_delay_timer >= ENEMY_SPAWN_FRAMES

    # --- Update enemies ---
    new_enemy_pos, new_enemy_vel, new_enemy_idx, new_enemy_timer, new_enemy_sides = enemy_step(
        state.enemy_pos,
        state.enemy_vel,
        state.enemy_platform_idx,
        state.enemy_timer,
        PLATFORMS,
        state.enemy_initial_sides,
        active_mask
    )

    new_delay_timer = state.enemy_delay_timer + 1

    new_pow_hits = jnp.where(pow_hit, jnp.minimum(state.pow_hits + 1, 3), state.pow_hits)

    return GameState(
        pos=jnp.array([new_x, new_y]),
        vel=jnp.array([vx, vy_final]),
        on_ground=landed,
        jump_phase=jump_phase.astype(jnp.int32),
        ascend_frames=asc_left.astype(jnp.int32),
        enemy_pos=new_enemy_pos,
        enemy_vel=new_enemy_vel,
        enemy_platform_idx=new_enemy_idx,
        enemy_timer=new_enemy_timer,
        enemy_initial_sides=new_enemy_sides,
        enemy_delay_timer=new_delay_timer,
        pow_hits=new_pow_hits
    )


# -------------------- Enemy step function -----------------
@jit
def enemy_step(enemy_pos, enemy_vel, enemy_platform_idx, enemy_timer, platforms, initial_sides, active_mask):
    ew, eh = ENEMY_SIZE
    TOP_PLATFORMS = jnp.array([1, 2], dtype=jnp.int32)  # indices of top platforms
    ENEMY_TOP_START_IDX = jnp.array([1, 2], dtype=jnp.int32)  # which top platform each enemy starts from

    def platform_below_at(x_pos, current_y):
        px, py, pw, ph = platforms[:, 0], platforms[:, 1], platforms[:, 2], platforms[:, 3]
        left = px
        right = px + pw
        supported_x = (x_pos + ew > left) & (x_pos < right)
        below_y = py > current_y
        candidates = supported_x & below_y
        return jnp.where(candidates, py, jnp.inf)

    def step_one(pos, vel, p_idx, timer, side, i):
        x, y = pos
        vx, vy = vel
        platform = platforms[p_idx]
        plat_x, plat_y, plat_w, _ = platform
        plat_left = plat_x
        plat_right = plat_x + plat_w

        new_x = x + vx

        walking_off_left = new_x < plat_left
        walking_off_right = new_x + ew > plat_right
        walking_off_edge = walking_off_left | walking_off_right

        fall_x = jnp.where(walking_off_left, plat_left - ew, plat_right)

        platforms_below_y = platform_below_at(fall_x, y + eh)
        min_platform_below_y = jnp.min(platforms_below_y)
        has_platform_below = min_platform_below_y != jnp.inf

        platform_edges_x = jnp.array([plat_left, plat_right - ew])
        platforms_below_any_left = jnp.min(platform_below_at(platform_edges_x[0], y + eh))
        platforms_below_any_right = jnp.min(platform_below_at(platform_edges_x[1], y + eh))
        is_lowest_platform = (platforms_below_any_left == jnp.inf) & (platforms_below_any_right == jnp.inf)

        timer = jnp.asarray(timer + 1, dtype=jnp.int32)

        def teleport_down(pos_x, vx, side, ew, eh, platforms, min_platform_below_y):
            # Mask of platforms at the target Y level
            same_y_mask = platforms[:, 1] == min_platform_below_y

            # Find platforms under the current x
            plat_lefts = platforms[:, 0]
            plat_rights = platforms[:, 0] + platforms[:, 2]

            inside_x_mask = (pos_x >= plat_lefts) & (pos_x <= plat_rights)
            valid_mask = same_y_mask & inside_x_mask

            # Choose first matching platform index or fallback
            horizontal_distance = jnp.where(valid_mask, 0.0, 1e6)
            idx_below = jnp.argmin(horizontal_distance)

            new_pos_x = jnp.array(pos_x, dtype=jnp.float32)
            new_pos_y = jnp.array(platforms[idx_below, 1] - eh, dtype=jnp.float32)

            new_vx_down = vx  # Preserve horizontal velocity
            new_timer = jnp.array(0, dtype=jnp.int32)
            new_side = jnp.array(side, dtype=jnp.int32)

            return (new_pos_x, new_pos_y, new_vx_down, idx_below, new_timer, new_side)

        def teleport_up(side, target_top_idx):
            def pos_and_vel_for_top(idx):
                left_side_x = platforms[idx, 0] + 5.0
                right_side_x = platforms[idx, 0] + platforms[idx, 2] - ew - 5.0
                start_x = jnp.where(side == 1, right_side_x, left_side_x)
                start_vx = jnp.where(side == 1, -0.5, 0.5)
                start_y = platforms[idx, 1] - eh
                return (start_x.astype(jnp.float32), start_y.astype(jnp.float32), start_vx.astype(jnp.float32))

            new_pos_x, new_pos_y, new_vx_up = pos_and_vel_for_top(target_top_idx)
            new_timer = jnp.array(0, dtype=jnp.int32)
            new_side = jnp.array(side, dtype=jnp.int32)

            return (new_pos_x, new_pos_y, new_vx_up, target_top_idx, new_timer, new_side)

        new_x_final = new_x
        new_y_final = y
        new_vx_final = vx
        new_p_idx = p_idx
        new_side = side

        # Teleport down if walking off edge and platform below exists
        new_x_final, new_y_final, new_vx_final, new_p_idx, timer, new_side = jax.lax.cond(
            walking_off_edge & has_platform_below,
            lambda _: teleport_down(new_x, vx, side, ew, eh, platforms, min_platform_below_y),
            lambda _: (new_x, y, vx, p_idx, timer, side),
            operand=None
        )

        TELEPORT_WAIT = 100

        def wait_and_teleport_up():
            def do_teleport():
                # Use enemy index i to pick correct top platform start idx
                return teleport_up(side, ENEMY_TOP_START_IDX[i])

            def keep_patrol():
                return (new_x_final, new_y_final, -new_vx_final, new_p_idx, timer, new_side)

            return jax.lax.cond(timer >= TELEPORT_WAIT, do_teleport, keep_patrol)

        # Teleport up condition: walking off edge, no platform below, and on lowest platform
        new_x_final, new_y_final, new_vx_final, new_p_idx, timer, new_side = jax.lax.cond(
            walking_off_edge & (~has_platform_below) & is_lowest_platform,
            wait_and_teleport_up,
            lambda: (new_x_final, new_y_final, new_vx_final, new_p_idx, timer, new_side)
        )

        # Reverse direction if walking off edge but not on lowest platform and no platform below
        def reverse_dir():
            return (new_x_final, new_y_final, -new_vx_final, new_p_idx, timer, new_side)

        new_x_final, new_y_final, new_vx_final, new_p_idx, timer, new_side = jax.lax.cond(
            walking_off_edge & (~has_platform_below) & (~is_lowest_platform),
            reverse_dir,
            lambda: (new_x_final, new_y_final, new_vx_final, new_p_idx, timer, new_side)
        )

        return (jnp.array([new_x_final, new_y_final]),
                jnp.array([new_vx_final, 0.0]),
                new_p_idx,
                timer,
                new_side)

    def conditional_step(pos, vel, idx, timer, side, i, active):
        return jax.lax.cond(
            active,
            lambda _: step_one(pos, vel, idx, timer, side, i),
            lambda _: (pos, vel, idx, timer, side),
            operand=None
        )

    # Prepare index array to pass enemy indices to step_one
    indices = jnp.arange(enemy_pos.shape[0])

    new_pos, new_vel, new_idx, new_timer, new_sides = jax.vmap(
        conditional_step, in_axes=(0, 0, 0, 0, 0, 0, 0)
    )(enemy_pos, enemy_vel, enemy_platform_idx, enemy_timer, initial_sides, indices, active_mask)

    return new_pos, new_vel, new_idx, new_timer, new_sides


# -------------------- MAIN ----------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH * WINDOW_SCALE, SCREEN_HEIGHT * WINDOW_SCALE)
    )
    pygame.display.set_caption("JAX Mario Bros Prototype")
    clock = pygame.time.Clock()

    state = init_state()
    running = True

    # Lives and game over flag
    lives = 3
    game_over = False

    # -------- pattern management ----------------------------------
    movement_pattern = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
    pat_len = len(movement_pattern)
    idx_right = 0  # current index if RIGHT is held
    idx_left = 0  # current index if LEFT is held
    jump = 0
    jumpL = False
    jumpR = False
    # --- New Variables for stop frames -------------------------
    last_dir = 0
    brake_frames_left = 0
    BRAKE_DURATION = 10  # in Frames
    BRAKE_TOTAL_DISTANCE = 7.0  # in Pixels
    brake_speed = BRAKE_TOTAL_DISTANCE / BRAKE_DURATION  # ≈ 0.7 px/frame


    def draw_rect(color, rect):
        r = pygame.Rect(rect)
        r.x *= WINDOW_SCALE
        r.y *= WINDOW_SCALE
        r.w *= WINDOW_SCALE
        r.h *= WINDOW_SCALE
        pygame.draw.rect(screen, color, r)

    font = pygame.font.SysFont(None, 36)  # Font for lives/game over

    while running:
        # ------------------- INPUT --------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:  # stops game when Escape is pressed
            running = False

        if game_over:
            screen.fill((0, 0, 0))
            game_over_text = font.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH * WINDOW_SCALE // 2, SCREEN_HEIGHT * WINDOW_SCALE // 2))
            screen.blit(game_over_text, text_rect)
            pygame.display.flip()
            continue  # Skip the rest of loop to freeze game

        if state.on_ground:
            move = 0.0
            jump = 0
            jumpL = False
            jumpR = False

        if not jumpL and not jumpR:
            pressed_right = keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]
            pressed_left = keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]

        if keys[pygame.K_SPACE]:
            jump = 1

        if jump == 0:
            if pressed_right:
                move = movement_pattern[idx_right]
                idx_right = (idx_right + 1) % pat_len
                idx_left = 0
                last_dir = 1
                brake_frames_left = 0
            elif pressed_left:
                move = -movement_pattern[idx_left]
                idx_left = (idx_left + 1) % pat_len
                idx_right = 0
                last_dir = -1
                brake_frames_left = 0
            else:
                if last_dir != 0 and brake_frames_left == 0 and state.on_ground:
                    brake_frames_left = BRAKE_DURATION

                if brake_frames_left > 0:
                    move = last_dir * brake_speed
                    brake_frames_left -= 1
                    if brake_frames_left == 0:
                        last_dir = 0
                else:
                    move = 0.0
        elif jump == 1:
            if pressed_right or jumpR:
                move = movement_pattern[idx_right]
                idx_right = (idx_right + 1) % pat_len
                idx_left = 0
                jumpR = True
            elif pressed_left or jumpL:
                move = -movement_pattern[idx_left]
                idx_left = (idx_left + 1) % pat_len
                idx_right = 0
                jumpL = True
            brake_frames_left = 0
            last_dir = 0

        # ----------------- UPDATE & RENDER ------------------------
        state = step(state, jnp.array([move, jump], dtype=jnp.float32))

        screen.fill((0, 0, 0))
        # player
        if brake_frames_left > 0:
            draw_rect((255, 0, 0), (*state.pos.tolist(), *PLAYER_SIZE))
        else:
            draw_rect(PLAYER_COLOR, (*state.pos.tolist(), *PLAYER_SIZE))
        # platforms
        for plat in PLATFORMS:
            draw_rect((228, 111, 111), plat.tolist())
        # POW block
        if state.pow_hits < 3:
            draw_rect((201, 164, 74), POW_BLOCK[0].tolist())

        # Only draw active enemies (spawned already)
        active_enemies = [
            ep for i, ep in enumerate(state.enemy_pos)
            if state.enemy_delay_timer[i] >= ENEMY_SPAWN_FRAMES[i]
        ]
        for ep in active_enemies:
            draw_rect((255, 0, 0), (*ep.tolist(), *ENEMY_SIZE))

        # Collision check for active enemies only (basic death)
        if check_enemy_collision(state.pos, jnp.array(active_enemies)):
            lives -= 1
            if lives <= 0:
                game_over = True
            else:
                print(f"Hit by enemy! Lives left: {lives}. Respawning...")
                state = reset()

        # Draw lives on screen
        lives_text = font.render(f"Lives: {lives}", True, (228, 111, 111))  # Yellow color
        text_rect = lives_text.get_rect(center=(SCREEN_WIDTH * WINDOW_SCALE // 2, 30))
        screen.blit(lives_text, text_rect)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()