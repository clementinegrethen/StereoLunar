local degToRad = 3.141592265358 / 180
local moon_radius = 1737400.0

projection = 'stereo.txr'

params = {
    MOON_AVG_RADIUS  = moon_radius,
    inv_scale = {1/5.0, -1/5.0},
    south = True,
    translation = {-75840.0, 75840.0 },
}
