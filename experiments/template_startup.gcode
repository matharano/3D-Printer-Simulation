;FLAVOR:Marlin
;TIME:265
;Filament used: 0.113878m
;Layer height: 0.2
;MINX:60
;MINY:20
;MINZ:0.2
;MAXX:120
;MAXY:30
;MAXZ:0.6
;Generated with Cura_SteamEngine main
M82 ;absolute extrusion mode
; Ender 3 Custom Start G-code
G92 E0 ; Reset Extruder
G28 ; Home all axes
M104 S175 ; Start heating up the nozzle most of the way
M190 S60 ; Start heating the bed, wait until target temperature reached
M109 S{$TEMPERATURE} ; Finish heating the nozzle
G1 Z2.0 F3000 ; Move Z Axis up little to prevent scratching of Heat Bed
G1 X0.1 Y20 Z0.3 F5000.0 ; Move to start position
G1 X0.1 Y200.0 Z0.3 F1500.0 E15 ; Draw the first line
G1 X0.4 Y200.0 Z0.3 F5000.0 ; Move to side a little
G1 X0.4 Y20 Z0.3 F1500.0 E30 ; Draw the second line
G92 E0 ; Reset Extruder
G1 Z2.0 F3000 ; Move Z Axis up little to prevent scratching of Heat Bed
G1 X5 Y20 Z0.3 F5000.0 ; Move over to prevent blob squish
G92 E0
G92 E0
G1 F1500 E-6.5
;LAYER_COUNT:{$LAYER_COUNT}
;LAYER:0
M107
M106 S85
M106 P1 S85
;TYPE:WALL-INNER
G1 F1500 E0
