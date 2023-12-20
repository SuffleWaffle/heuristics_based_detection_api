def decimal_to_rgb(decimal_color):
    red = (decimal_color >> 16) & 255
    green = (decimal_color >> 8) & 255
    blue = decimal_color & 255
    return [red, green, blue]

def hex_to_rgb(hex_color):
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return [red, green, blue]