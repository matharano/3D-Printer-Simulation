;TIME_ELAPSED:0
M107
G91 ;Relative positioning
G1 E-2 F2700 ;Retract a bit
G1 E-2 Z0.2 F2400 ;Retract and raise Z
G1 X5 Y5 F3000 ;Wipe out
G1 Z10 ;Raise Z more
G90 ;Absolute positioning

G1 X0 Y220 ;Present print
M106 S0 ;Turn-off fan
M106 P1 S0 ;Turn-off fan
M104 S0 ;Turn-off hotend
M140 S0 ;Turn-off bed

M84 X Y E ;Disable all steppers but Z

M82 ;absolute extrusion mode
M104 S0
;End of Gcode
;SETTING_3 {"global_quality": "[general]\\nversion = 4\\nname = Standard Quality
;SETTING_3  #2\\ndefinition = creality_ender3pro\\n\\n[metadata]\\ntype = qualit
;SETTING_3 y_changes\\nquality_type = standard\\nsetting_version = 20\\n\\n[valu
;SETTING_3 es]\\nadhesion_type = none\\n\\n", "extruder_quality": ["[general]\\n
;SETTING_3 version = 4\\nname = Standard Quality #2\\ndefinition = creality_ende
;SETTING_3 r3pro\\n\\n[metadata]\\ntype = quality_changes\\nquality_type = stand
;SETTING_3 ard\\nsetting_version = 20\\nposition = 0\\n\\n[values]\\ninfill_spar
;SETTING_3 se_density = 100\\n\\n"]}