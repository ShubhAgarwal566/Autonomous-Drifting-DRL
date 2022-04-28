import csv
import matplotlib.pyplot as plt
import math
import numpy as np
file = open("example_waypoints.csv")
# Python program to read image using OpenCV
 
# importing OpenCV(cv2) module
import cv2
 
# Save image in set directory
# Read RGB image
# img = cv2.imread('outputs/final_map.pgm')
 
# Output img with window name as 'image'
# cv2.imshow('image', img)

origin = -11.8, -5.3
 
csvreader = csv.reader(file)

header = next(csvreader)
# print(header)
rows = []
f = open('/home/jet/Desktop/waypoints_beta.csv', 'w')
# f1 = open('inputs/tracks/race3sat.csv')


# csvreader_width = csv.reader(f1)
writer = csv.writer(f)
xlist = []
ylist = []
vellist = []
headinglist = [] 
prev_heading = 0.0
op = 1.0
for i,row in enumerate(csvreader):
	if i>=2:
		row =row[0].split(";")
		# break
		x,y,heading, vel  = float(row[1]),float(row[2]), float(row[3]), float(row[5])#, float(row[7])
		xlist.append(x)
		ylist.append(y)
		vellist.append(vel)
		headinglist.append(heading)

		#switch directions
		beta = heading+op*0.15 
		if heading*prev_heading <0:
			# print(heading)
			# print(prev_heading)
			plt.scatter(x,y)
			print("iter",i)
			# print(x,y)
			op = op * -1
			print("heading", heading)
			print("beta",beta)

		prev_heading = heading


		# rows.append(row)
		writer.writerow([x,y,heading,vel,beta])

print("Mean:", sum(vellist)/len(vellist))

# for i,row in enumerate(csvreader_width):
# 	# if i>=2:
# 	# row =row[0].split(";")
# 	# print(row)
# 	# break
# 	rw,lw = float(row[2]),float(row[3])
	
# 	m = xlist[i] - xlist[i+1] / ylist[i+1] - ylist[i]
# 	#perpendicular slope
# 	x_off = rw*math.sqrt(1/(1+m**2))
# 	y_off = m*rw*math.sqrt(1/(1+m**2))


# 	xl_off = lw*math.sqrt(1/(1+m**2))
# 	yl_off = m*lw*math.sqrt(1/(1+m**2))

# 	xright.append(x_off)
# 	yright.append(y_off)

# 	xleft.append(xl_off)
# 	yleft.append(yl_off)




	# rows.append(row)
	# writer.writerow([x,y,vel])

# plt.imshow(img)
# plt.plot(xright,yright)
# plt.plot(xleft,yleft)

plt.plot(xlist,ylist)
plt.show()
f.close()
file.close()