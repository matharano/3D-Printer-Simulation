# TODO:
# 1. Functions for each of the defects
#   a. ABC
#   b. Over extrusion
#   c. Under extrusion
#   d. Void
#   e. Geometric deviation
from abc import ABC, abstractmethod

from .parser import Command, Gcode

class Defect(ABC):
    def __init__(self, incidence_ratio:float, intensity:float) -> None:
        """Parameters:
        -----
        *  @param incidence_ratio ∈ [0, 1]: percentage of the gcode commands upon which the defect may incide\n
        *  @param intensity ∈ [0, 1]: percentage of the intensity of the defect"""
        super().__init__()
        assert 0 <= incidence_ratio <= 1, "incidence_ratio should be a float between 0 and 1"
        assert 0 <= intensity <= 1, "incidence_ratio should be a float between 0 and 1"

        self.incidence_ratio = incidence_ratio
        self.intensity = intensity
    
    @abstractmethod
    def apply(self, command:Command) -> None:
        pass

class OverExtrusion(Defect):
    def apply(self, command:Command) -> None:
        if not (type(command) == float or type(command) == float): return
        last_coords = command.get_last_position()
        original_extrusion = command.e - last_coords['e']
        command.e += original_extrusion * self.intensity

if __name__ == '__main__':
    gcode = Gcode('data/defects_scans/single_line_cross/no_defect/CE3PRO_single_line_cross.gcode')
    defects = [OverExtrusion(1, 1)]
    gcode.apply_defects(defects)
    gcode.save('./source/gcode/gcode_with_defects.gcode')