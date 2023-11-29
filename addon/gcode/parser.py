from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING
import os, random

import logging
log = logging.getLogger("SmoPa3D")

if TYPE_CHECKING:
    from .defects import Defect

@dataclass
class GcodeCommand:
    gcode: Gcode
    id: int
    command_line: Optional[str] = None
    m: Optional[int] = None
    s: Optional[int] = None
    g: Optional[int] = None
    f: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    e: Optional[float] = None

    def __post_init__(self) -> None:
        assert sum([self.command_line is not None, self.g is not None, self.m is not None]) == 1, "Either command_line or command must be input to Command class"
        self.command_list = ['M', 'S', 'G', 'F', 'X', 'Y', 'Z', 'E']
        if self.command_line[-1] == '\n': self.command_line = self.command_line[:-1]

        if self.command_line is not None:
            for arg in self.command_line.split(' '):
                if arg[0] == ';': break
                if arg[0] in self.command_list:
                    self.__dict__[arg[0].lower()] = float(arg[1:]) if len(arg[1:]) > 0 else ''

    def __str__(self) -> str:
        if not self.is_transformable: return self.command_line
        parameters = []
        for arg in self.command_list:
            value = self.__dict__[arg.lower()]
            if value is None:
                continue
            elif type(value) == str:
                parameters.append(value)
            else:
                parameters.append(arg + str(round(value, 5)))
        return ' '.join(parameters)

    def format(self) -> str:
        return self.__str__()

    @property
    def last_position(self, parameters_to_search:list[str]=['x', 'y', 'z', 'e']) -> dict:
        """Locate the last ocurrence of coordinates x, y, z, e in parent gcode. If no previous ocurrence is found, returns `None`\n
        ```
        return {x: float, y: float, z: float, e: float}
        ```"""
        coords = {param: None for param in parameters_to_search}
        for i in range(self.id-1, 0, -1):
            if all([pos is not None for pos in coords.values()]):
                break
            for coord in coords.keys():
                if self.gcode.commands[i].__dict__[coord] is not None and self.gcode.commands[i].__dict__[coord] != '' and coords[coord] is None:
                    coords[coord] = self.gcode.commands[i].__dict__[coord]
        return coords
    
    def last_ocurrence(self, parameter:str) -> Optional[float]:
        """Locate the last ocurrence of the parameter in parent gcode. If no previous ocurrence is found, returns `None`"""
        if parameter.upper() not in self.command_list:
            log.warning(f'No previous ocurrence of parameter {parameter} not found in command list')
            return None
        
        for i in range(self.id-1, 0, -1):
            if self.gcode.commands[i].__dict__[parameter] is not None and self.gcode.commands[i].__dict__[parameter] != '':
                return self.gcode.commands[i].__dict__[parameter]
    
    @property
    def is_transformable(self) -> bool:
        """Check if command is transformable, i.e., if it:\n
        * is not part of printer setup
        * is not part of printer unset
        * is not a comment
        * is not M command
        * has moved
        * has extruded
        """
        is_setup = self.id <= self.gcode.setup_end
        is_unset = self.id >= self.gcode.unset_start
        is_comment = self.command_line[0] == ';'
        is_m_command = self.m is not None
        has_moved = sum([self.x is not None, self.y is not None, self.z is not None]) > 0
        has_extruded = self.e is not None
        return not any([is_setup, is_unset, is_comment, is_m_command, not has_moved, not has_extruded])
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        self.__dict__[__name] = __value


class Gcode:
    def __init__(self, path:str) -> None:
        assert os.path.exists(path), f"File path {os.path.abspath(path)} does not exist"
        self.path = path

        # Read gcode
        with open(path, 'r') as gcodefile:
            gcode = gcodefile.readlines()
        self.gcode = gcode
        self.setup_end = next(i for (i, row) in enumerate(self.gcode) if ';LAYER_COUNT' in row)
        self.unset_start = next(i for i in range(len(self.gcode)-1, 0, -1) if ';TIME_ELAPSED' in self.gcode[i])
        self.commands = [GcodeCommand(self, command_line=command, id=i) for (i, command) in enumerate(gcode) if len(command.replace('\n', '')) > 0]

    
    def apply_defects(self, defectset:list[Defect], overlap:bool=False) -> None:
        """Transform gcode by applying randomly the set of defects. Together with the transformed gcode, it outputs a label in json format indicating the coordinates of each of the synthetized defects.\n
        Parameters:
        -----
        * @param overlap: if false, each command can contain a maximum of 1 defect. If True, more than one defect can happen in each command\n"""
        
        # if not overlap: assert if sum is equal or less than 1
        transformable = [command for command in self.commands if command.is_transformable]
        total_commands = len(transformable)
        for defect in defectset:
            sample = random.sample(transformable, int(defect.incidence_ratio * total_commands))
            for command in sample:
                if not overlap:
                    transformable.remove(command)
                command = defect.apply(command)

    def save(self, path:str=None) -> None:
        if path is None:
            path = self.path.replace('.gcode', '_with_defects.gcode')
        
        with open(path, 'w') as fle:
            for command in self.commands:
                fle.write(command.format() + '\n')