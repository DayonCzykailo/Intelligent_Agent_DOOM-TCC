
import os
from random import choice
from time import sleep

import vizdoom as vzd


if __name__ == "__main__":
    # Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))
    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)  # Bullet holes and blood on the walls
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  # Like smoke and blood
    game.set_render_messages(False)  # In-game text messages
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
    game.set_available_game_variables([vzd.GameVariable.AMMO2])
    game.set_episode_timeout(200)
    game.set_episode_start_time(10)
    game.set_window_visible(True)
    game.set_living_reward(-1)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()


    actions = [[True, False, False], [False, True, False], [False, False, True]]

    episodes = 10
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    for i in range(episodes):
        print(f"Episode #{i + 1}")

        game.new_episode()

        while not game.is_episode_finished():

            # Gets the state
            state = game.get_state()

            # Which consists of:
            n = state.number
            vars = state.game_variables

            # Different buffers (screens, depth, labels, automap, audio)
            # Expect of screen buffer some may be None if not first enabled.
            screen_buf = state.screen_buffer
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            audio_buf = state.audio_buffer

            # List of labeled objects visible in the frame, may be None if not first enabled.
            labels = state.labels

            # List of all objects (enemies, pickups, etc.) present in the current episode, may be None if not first enabled
            objects = state.objects

            # List of all sectors (map geometry), may be None if not first enabled.
            sectors = state.sectors

            r = game.make_action(choice(actions))


            if sleep_time > 0:
                sleep(sleep_time)

        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")

    game.close()