# -*- coding: utf-8 -*-
"""
@author: grh

Project: SmoPa3D

Description: This code is used for the manipulation of the gcode in order to generate slices
"""
import pandas as pd
from random import sample, randint
import re
import numpy as np
from scipy.interpolate import interp1d
import os

#===============================================================================
## Generating of gcode slices in order to send them to the printer
#=============================================================================== 
def convertGcode2Slices(rel_path = "/data/printjobs/test_part.gcode"):

    with open(os.path.join(os.environ['ROOT'], rel_path), "r") as gcode:   
        data = gcode.read() ### read in the G-Code

        ##Get the layer height  
        layerheight_start = re.search(";Layer height: ", data).end()
        layerheight_end = re.search(";MINX:", data).start()
        layerheight = float(data[layerheight_start:layerheight_end])
        
        ##Get the number of Layers    
        numberOfLayers_start = re.search(";LAYER_COUNT:", data).end()
        numberOfLayers_end = re.search(";LAYER:0", data).start()
        numberOfLayers = int(data[numberOfLayers_start:numberOfLayers_end])
        numberOfLayers = numberOfLayers - 1 ## subtracting -1 in order to get the real number of layers (the counting starts at 0 -> for example if 169 layers are displayed, the last layer ist layer 168)        

        ##Get YMin und YMax for to select the measuring area
        #Getting YMin
        numberOfLayers_start = re.search(";MINY:", data).end()
        numberOfLayers_end = re.search(";MINZ:", data).start()
        YMin = float(data[numberOfLayers_start:numberOfLayers_end])

        #Getting YMax
        numberOfLayers_start = re.search(";MAXY:", data).end()
        numberOfLayers_end = re.search(";MAXZ:", data).start()
        YMax = float(data[numberOfLayers_start:numberOfLayers_end])
        
        ##Seperate the first Layer from the G-Code
        numberOfLayers_end = re.search(";LAYER:1", data).start()
        slices = np.array(str(data[:numberOfLayers_end]))

        i = 1 ## starting the counter at 1 because the 0 layer was already seperated
        while i <= numberOfLayers:
            ## Select start and end Layer (code is searching for ";LAYER:i" and "LAYER:i+1" in order to seperate each layer)
            startLayer = re.search(";LAYER:" + str(i), data).start() ## getting the number of the character of the current layer in order to extract the layer
            
            ## Seperate the other Layers from the G-Code
            if i < numberOfLayers:
                endLayer = re.search(";LAYER:" + str(i+1), data).start()  ## getting the number of the character of the next layer in order to extract the layer
                slices = np.append(slices, (str(data[startLayer:endLayer]))) ## seperating the layer and appending it to the numpy array

            ## Seperate the last Layer from the G-Code
            else:
                slices = np.append(slices, (str(data[startLayer:]))) ## seperating the layer and appending it to the numpy array
            i += 1

    return numberOfLayers, layerheight, YMin, YMax, slices

#===============================================================================
## Generating of gcode with randomly induced defects (pores)
#=============================================================================== 
def convertGcode2SlicesWithDefects(rel_path = "/data/printjobs/test_part.gcode", output_path = "/data/printjobs/test_part_defective.gcode"):

    with open(os.path.join(os.environ['ROOT'], rel_path), "r") as gcode:     
        data = gcode.read() ### read in the G-Code

        ##Get the layer height  
        layerheight_start = re.search(";Layer height: ", data).end()
        layerheight_end = re.search(";MINX:", data).start()
        layerheight = float(data[layerheight_start:layerheight_end])
        
        ##Get the number of Layers    
        numberOfLayers_start = re.search(";LAYER_COUNT:", data).end()
        numberOfLayers_end = re.search(";LAYER:0", data).start()
        numberOfLayers = int(data[numberOfLayers_start:numberOfLayers_end])
        numberOfLayers = numberOfLayers - 1 ## subtracting -1 in order to get the real number of layers (the counting starts at 0 -> for example if 169 layers are displayed, the last layer ist layer 168)        

        ##Get YMin und YMax for to select the measuring area
        #Getting YMin
        numberOfLayers_start = re.search(";MINY:", data).end()
        numberOfLayers_end = re.search(";MINZ:", data).start()
        YMin = float(data[numberOfLayers_start:numberOfLayers_end])

        #Getting YMax
        numberOfLayers_start = re.search(";MAXY:", data).end()
        numberOfLayers_end = re.search(";MAXZ:", data).start()
        YMax = float(data[numberOfLayers_start:numberOfLayers_end])
        
    with open(os.path.join(os.environ['ROOT'], rel_path), "r") as gcode:
        ## Convert gcode into a Pandas Dataframe in order to process it
        lines = gcode.readlines()
        lines = [line.strip() for line in lines]
        df = pd.Series(lines)
        
        ## get the locations in the gcode, which contain those strings in order to seperate the code regarding the type which is printed
        locations = df.loc[df.str.contains(";MESH:NONMESH|;TYPE:SKIN|;TYPE:FILL|;TYPE:WALL-INNER|;TYPE:WALL-OUTER", case= False)].index.values

        i = 0
        ## sets the resolution/size of the defects in mm
        resolution = 1
        ## go through each location. If its SKIN oder FILL, defects will be generated

        while i < (len(locations)-1):
            if df[locations[i]] == ";TYPE:SKIN" or df[locations[i]] == ";TYPE:FILL":
                
                ## get each for each command block the commands in which material is extruded
                position_commands = df[(locations[i]+2):locations[i+1]].loc[df.str.contains("G1", case= False)].index.values
                position_commands = df[position_commands].loc[df.str.contains("X", case= False)].index.values
                position_commands = df[position_commands].loc[df.str.contains("E", case= False)].index.values
                
                
                ## creates random numbers in range of the len of the position commands in order get a random entry from position commands
                len_commands = len(position_commands)
                random_number_array = sample(range(0, len_commands), int(len_commands/20))
                random_number_array.sort(reverse=True)

                ## see if there are any random numbers created
                if random_number_array != []:
                    
                    ## go throug each printing section (FILL or SKIN) and create defects
                    j = 0
                    while j < int(len_commands/20):
                        ## Gets the x-position, y-position and e-position in order to get the starting point of the considered command
                        ## Get X
                        X_start = re.search("X", df[position_commands[random_number_array][j]]).end()
                        X_end = re.search("Y", df[position_commands[random_number_array][j]]).start()
                        x = float(df[position_commands[random_number_array][j]][X_start:X_end])
                        ## Get Y
                        Y_start = re.search("Y", df[position_commands[random_number_array][j]]).end()
                        Y_end = re.search("E", df[position_commands[random_number_array][j]]).start()
                        y = float(df[position_commands[random_number_array][j]][Y_start:Y_end])
                        ## Get E
                        E_start = re.search("E", df[position_commands[random_number_array][j]]).end()
                        e = float(df[position_commands[random_number_array][j]][E_start:])
                        ## Create a coordinate variable
                        coordinate = np.array((x,y))
                
                        ## Gets the previous x-position and y-position in order to get the starting point of the previous command
                        ## Get X
                        X_start = re.search("X", df[position_commands[random_number_array][j]-1]).end()
                        X_end = re.search("Y", df[position_commands[random_number_array][j]-1]).start()
                        prev_x = float(df[position_commands[random_number_array][j]-1][X_start:X_end])
                        ## Get Y
                        Y_start = re.search("Y", df[position_commands[random_number_array][j]-1]).end()
                        if re.search("E", df[position_commands[random_number_array][j]-1]) is not None: 
                            Y_end = re.search("E", df[position_commands[random_number_array][j]-1]).start()
                            prev_y = float(df[position_commands[random_number_array][j]-1][Y_start:Y_end])
                        else:
                            prev_y = float(df[position_commands[random_number_array][j]-1][Y_start:])
                        ## Create a coordinate variable
                        prev_coordinates = np.array((prev_x,prev_y))
                    
                        ## gets the total distance between the considered commands ("coordinates" and "prev_coordinates")
                        dist_total = np.linalg.norm(coordinate-prev_coordinates)
                        
                        ## if the total distance is lower than 2mm the command will be executed but no filament is extruded -> the defect is then max 2 mm long
                        if dist_total < 2:
                            df[position_commands[random_number_array][j]] = "G1 X" + str(x) + " Y" + str(y) + " \nG92 E" + str(e)

                        ## if the total distance is higher than 2mm the "line" will be interupted on a random position and a defect will be inserted (nearly 1 mm long)
                        elif dist_total > 2:
                            
                            ## Interpolation of the previous coordinates (from the previous command) and the current coordinates (from the considered command)
                            prev_e = e - (dist_total * 0.033) ## 0.033 is the distance the E-axis is moving per mm
                            increment = int(dist_total/resolution) ## total number of points which are created in the interpolated array
                            x_interplt=[prev_x,x]
                            y_interplt=[prev_y,y]
                            f=interp1d(x_interplt,y_interplt)
                            x_coordinates = np.linspace(prev_coordinates[0],coordinate[0],increment)
                            y_coordinates = f(x_coordinates)                        

                            ## creates a random number in order to select on position in the created "line" of the interpolation
                            random_number = randint(1, (len(x_coordinates)-1))

                            ## gets the coordinate before the "defect"
                            coordinate_before_defect = np.array((x_coordinates[random_number-1], y_coordinates[random_number-1]))

                            ## gets the coordinate after the "defect"
                            coordinate_after_defect = np.array((x_coordinates[random_number], y_coordinates[random_number]))

                            ## gets the distance between the previous coordinate and the coordinate before the "defect" in order to calculate the extrusion
                            dist_1 = np.linalg.norm(prev_coordinates-coordinate_before_defect)
                            e_value_1 = round(prev_e + (0.033 * dist_1),5)

                            ## gets the distance between the the coordinate before the "defect" and the coordinate after the "defect" in order to calculate the "extrusion"
                            dist_2 = np.linalg.norm(coordinate_before_defect-coordinate_after_defect)
                            e_value_2 = round(e_value_1 + (0.033 * dist_2),5)

                            ## if the random number is 1 and therefore the defect is just in the beginning of the line, the new code is generated
                            ## The defect in a 3 mm line looks like this |x--| (with - as printed normal and x as not printed and therefore a defect)
                            if random_number == 1:
                                df[position_commands[random_number_array][j]] = "G0 X" + str(coordinate_after_defect[0]) + " Y" + str(coordinate_after_defect[1]) + " \nG92 E" + str(e_value_2) + " \n" + df[position_commands[random_number_array][j]]
                            
                            ## The defect in a 3 mm line looks like this |--x| (with - as printed normal and x as not printed and therefore a defect)
                            elif random_number == (len(x_coordinates)-1):
                                df[position_commands[random_number_array][j]] = "G1 X" + str(coordinate_before_defect[0]) + " Y" + str(coordinate_before_defect[1]) + " E" + str(e_value_1) + " \nG0 X" + str(x) + " Y" + str(y) + " \nG92 E" + str(e)
                            
                            ## The defect in a 3 mm line looks like this |-x-| (with - as printed normal and x as not printed and therefore a defect)
                            else:
                                df[position_commands[random_number_array][j]] = "G1 X" + str(coordinate_before_defect[0]) + " Y" + str(coordinate_before_defect[1]) + " E" + str(e_value_1) + " \nG0 X" + str(coordinate_after_defect[0]) + " Y" + str(coordinate_after_defect[1]) + " \nG92 E" + str(e_value_2) + " \n" + df[position_commands[random_number_array][j]]
                        j +=1
            i += 1

        ## Convert Pandas Dataframe back into str in order to be processed as gcode
        newlines = df.values.tolist()
        newgcode = "\n".join(newlines)
        
        ##Seperate the first Layer from the G-Code
        numberOfLayers_end = re.search(";LAYER:1", newgcode).start()
        slices = np.array(str(newgcode[:numberOfLayers_end]))

        i = 1 ## starting the counter at 1 because the 0 layer was already seperated
        while i <= numberOfLayers:
            ## Select start and end Layer (code is searching for ";LAYER:i" and "LAYER:i+1" in order to seperate each layer)
            startLayer = re.search(";LAYER:" + str(i), newgcode).start() ## getting the number of the character of the current layer in order to extract the layer
            
            ## Seperate the other Layers from the G-Code
            if i < numberOfLayers:
                endLayer = re.search(";LAYER:" + str(i+1), newgcode).start()  ## getting the number of the character of the next layer in order to extract the layer
                slices = np.append(slices, (str(newgcode[startLayer:endLayer]))) ## seperating the layer and appending it to the numpy array

            ## Seperate the last Layer from the G-Code
            else:
                slices = np.append(slices, (str(newgcode[startLayer:]))) ## seperating the layer and appending it to the numpy array
            i += 1
 
    #file  = open(os.path.join(os.environ['ROOT'],utput_path), "w")
    #file.write(newgcode)
    #file.close

    return numberOfLayers, layerheight, YMin, YMax, slices

if __name__ == '__main__':
    while True:
        try:
            convertGcode2SlicesWithDefects()
        except AttributeError:
            continue
        break