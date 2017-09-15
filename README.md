# Motivation: To be able to programmatically visualize the view from the top of a mountain or ridgeline.                
#                                                                                                                       
# Usage: Set the bounds, eye location, and direction in main.                                                           
# From the command line run this module. It will take a few minutes, and will produce 4 charts. 
# Click any point on the 'View' chart to see that location plotted on the elevation and relief maps.                    
#                                                                                                                       
# Approach: The current approach is inspired by the computer graphics technique raytracing. Basically we define         
# the location of an 'eye' and we decide on a canvas of actual locations. We then draw a ray from the                   
# eye to each point and follow it until it intersects with the terrain. We say thats what would be seen                 
# if a person were to look in that direction from that location.                                                        
#                                                                                                                       
# Definitions: As I built the raytracing I thought in terms of x, y, and z coordinates, but the elevation data          
# is based on latitude longitude and height in meters. Unfortunately this has yielded some tricky and messy code.       
# If I were to invest more time in this project I would want to simply this. As of now the conventions are that         
# latitude increases as you go North, longitude increases as you go West (sorry, this too is silly, but I was thinking  
# in terms of North America and started dropping the negative signs on the longtitude). Also worth checking the         
# order of arguments to functions, sometimes its lat, long, sometimes long, lat depending on context. Anytime x         
# is inappropriately used to refer to location its long and y is lat. When indicies are discussed however, due          
# to the way the elevation data is loaded (Here I probably could have just rotated the array, again apologies)          
# x increases as you go East and y increases as you go South. The arrays are indexed                                    
# [row][column] -> [y_index][x_index] yet another potential source of confusion.                                        
#                                                                                                                       
# Issues/todos: As of now this assumes a flat earth (http://rol.st/2xI43qR), but subtracts a certain amount from        
# each point depending on how far it is from the 'eye'. This is an approximation and I haven't done much to test        
# this against real world observations.                                                                                 
# The biggest source of error here is all distance <-> lat/long conversions are done at the eye. This means that        
# there is actually a slight curve to the rays as they get far from the eye. Hopefully this isn't too major a           
# distortion.                                                                                                           
# The proper solution to both would be to rethink everything in terms of three dimensional space, probably              
# declaring the origin as the center of the earth. This is a major overhaul and would really just entail starting       
# over. Hopefully this is still a decent approximation in most cases.       