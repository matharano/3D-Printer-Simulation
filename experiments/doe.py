"""Module to manage the design of experiments (DOE) to learn about the profile's cross-section"""

import os

def generate_experiment(folderpath:str="data/experiments/gcode", temperature:int=200, speed:int=1000, layer_count:int=3) -> None:
    """Generates a gcode file with the given temperature and speed settings"""
    # Setup
    os.makedirs(folderpath, exist_ok=True)
    gcode = setup(temperature=temperature, layer_count=layer_count)

    # Generate gcode
    global e
    e = 0
    y_start = 20
    length = 100
    for (layer_number, z) in enumerate([0.2, 0.4, 0.6]):
        gcode += f";LAYER:{layer_number}\n"
        direction = 1
        for i in range(20):
            gcode += print_strand(speed=speed, x=70 + i * 3, y=y_start + max(0, -direction * length), z=z, length=length * direction, feeding_rate=0.03 + i * 0.01)
            direction *= -1

    gcode += end()

    with open(f"{folderpath}/temperature_{temperature}_speed_{speed}.gcode", "w") as fle:
        fle.write(gcode)
    

def print_strand(speed:int, x:float, y:float, z:float, length:float, feeding_rate:float) -> str:
    """Returns a string to print a strand at the given coordinates"""
    global e
    e += abs(feeding_rate * length)
    y_final = y + length
    move_up = f"G0 F6000 Z{round(z + 0.4, 1)}\n"
    place_in_start = f"G0 F6000 X{round(x, 3)} Y{round(y, 3)}\n"
    move_down = f"G0 F6000 Z{round(z, 1)}\n"
    extrude = f"G1 F{round(speed, 3)} Y{round(y_final, 3)} E{round(e, 3)}\n"
    return move_up + place_in_start + move_down + extrude

def setup(temperature:int, layer_count:int) -> str:
    """Returns a string to setup the printer with the given temperature"""
    with open("doe/template_startup.gcode", "r") as fle:
        template = fle.read()
    
    template = template.replace("{$TEMPERATURE}", str(temperature))
    template = template.replace("{$LAYER_COUNT}", str(layer_count))
    return template

def end() -> str:
    """Returns a string to end the printing process"""
    with open("doe/template_end.gcode", "r") as fle:
        template = fle.read()
    return template

if __name__ == "__main__":
    def main():
        for temp in [180, 200]:
            for speed in [500, 1200]:
                generate_experiment(folderpath='data/experiments/gcode/v2', temperature=temp, speed=speed)
            
    main()