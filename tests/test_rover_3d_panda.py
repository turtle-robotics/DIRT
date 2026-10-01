from simulation.rover_3d_panda import (
    clamp_target_fps,
    lighting_profile,
    normalize_resolution,
    terrain_palette,
    weather_profile,
)


def test_normalize_resolution_handles_tuple_and_string():
    assert normalize_resolution((2560, 1440)) == (2560, 1440)
    assert normalize_resolution("1920x1080") == (1920, 1080)
    assert normalize_resolution(None) == (2560, 1440)


def test_clamp_target_fps_keeps_bounds():
    assert clamp_target_fps(90.0) == 90.0
    assert clamp_target_fps(20.0) == 60.0
    assert clamp_target_fps(180.0) == 120.0


def test_terrain_palette_and_lighting_profile_are_richer():
    low = terrain_palette("farmland", -0.20)
    high = terrain_palette("farmland", 0.70)
    assert high[1] > low[1]
    profile = lighting_profile("farmland")
    assert profile["sun_color"][0] > profile["fill_color"][0]
    assert profile["fog_density"] > 0.005
    weather = weather_profile("farmland")
    assert weather["mist_density"] > 0.010
    assert weather["sun_glow"][0] > weather["mist_color"][0]
