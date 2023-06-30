import numpy as np
length_grid=577
rows_grid=24  #sqrt(length_grid-1)
cols_grid=24
grid=np.arange(1, length_grid, 1, dtype=int)
grid=grid.reshape((cols_grid, rows_grid))
hrs= [0, 41, 50, 10, 15, 29, 5, 6, 51, 13, 11, 26, 7, 12, 43, 19, 8, 23, 3, 1, 2, 4, 9, 55]
fd_s= [0, 350, 682, 334, 762, 647, 1074, 663, 229, 865, 1142, 380, 418, 1197, 819, 909, 644, 995, 67, 723, 266, 650, 524, 94, 1093, 781, 777, 714, 1158, 552, 1240, 859, 886, 276, 809, 1036, 998, 997, 106, 546, 509, 858, 175, 1054, 822, 921, 1216, 125, 238, 1204, 457, 917, 1059, 400, 704, 322, 501, 780, 43, 1025, 372, 1003, 631, 1182, 582, 579, 247, 1176, 876, 883, 23, 710, 811, 940, 1041, 659, 844, 393, 512, 1, 494, 1232, 48, 994, 1157, 960, 407, 963, 543, 474, 701, 912, 1002, 689, 470, 890]

target_G = np.load('../../', allow_pickle=True).tolist() #run trainer file and put the value of target_G which contains actual
#origin and destination of  requeststarget_G = np.asarray(target_G, dtype=np.float32)
target_G = target_G.reshape(576, 576)
time_l=[]
folder=[]
for x in range(0,1247):
 time_instant=np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' +str(x) +'\\time.npy', allow_pickle=True).tolist()
 time_l.append(time_instant[0])
 folder.append(x)

import math
day=[]
for i in hrs:
    ax=math.floor( time_l[i]/(24*4))
    if ax==5 or ax==6 or ax==12 or ax==13 or ax==19 or ax==20 or ax==26 or ax==27:
      day.append('weekend')
    else:
        day.append('weekday')
print("total number of requests",np.sum(target_G))
# Map

import math
hour=[0]*len(time_l)
rounded_hour=[0]*len(time_l)
rounded_min=[0]*len(time_l)
day=[0]*len(time_l)
km=0
for i in time_l:
 hour[km]=math.ceil(i/4)
 rounded_hour[km]=hour[km]%24
 rounded_min[km] = hour[km] % (24*4)

 day[km]=math.ceil(hour[km]/(24*4))
 km=km+1



hr=[]
fd=[]
for i in time_l:
    if i in range(861,861+96):
      hr.append(i)
      fd.append(time_l.index(i))
hr_a=np.array(hr)
ars=np.argsort(hr_a)
hr_s=[]
fd_s=[]

for i in ars:
    hr_s.append(hr_a[i])
    fd_s.append(fd[i])


hrs=[]
res = [*set(rounded_hour)]
for i in range(0,len(time_l)):
     if math.floor( time_l[i]/(24*4))==14:
        hrs.append(1)

src_arr = []
dest_arr = []
reqs = []
c = 0
d_i = 0
c_i = 0
kd = 3

import math
relative = []
for i in range(0, cols_grid):
    for j in range(0, rows_grid):
        relative.append(tuple([i, j]))
positional = list(range(0, rows_grid * cols_grid + 1, 1))
map = zip(relative, positional)
map = dict(map)  # maps (0,0) to 0
print(map)
map1 = zip(positional, relative)  # maps 0 to (0,0)   i.e maps grid cells to coordinate
map1 = dict(map1)
print(map1)

src_arr = []
dest_arr = []
for i in range(0, rows_grid):
    for j in range(0, cols_grid):
        src_arr.append(i)
        dest_arr.append(j)


reqs_arr = []
for i in range(0, len(src_arr)):
    reqs_arr.append(target_G[src_arr[i]][dest_arr[i]])


def len_sp(source, dest):
    a, b = source
    c, d = dest
    shortest_path = [source]
    row = abs(c - a)  # move these many rows down
    col = abs(d - b)
    if c != a:
        row_dir = int(row / (c - a))
    else:
        row_dir = 1
    if d != b:
        col_dir = int(col / (d - b))
    else:
        col_dir = 1
    if row <= col:
        eql_move = row
 #we dont need to move any more rows now
    else:
        eql_move = col
    row = row - eql_move
    col = col - eql_move
    for i in range(1, eql_move + 1):  # adds (2,2),(3,3),...
        a = a + row_dir
        b = b + col_dir
        shortest_path.append(tuple([a, b]))

    for j in range(0, col):
        b = b + col_dir
        shortest_path.append(tuple([a, b]))
    for j in range(0, row):
        a = a + row_dir
        shortest_path.append(tuple([a, b]))

    length = len(shortest_path) - 1
    return length


src_arr = []
dest_arr = []
for i in range(0, 361):
    for j in range(0, 361):
        src_arr.append(i)
        dest_arr.append(j)


reqs_arr = []
for i in range(0, len(src_arr)):
    reqs_arr.append(target_G[src_arr[i]][dest_arr[i]])



def len_sp(source, dest):
    a, b = source
    c, d = dest
    shortest_path = [source]
    row = abs(c - a)  # move these many rows down
    col = abs(d - b)
    if c != a:
        row_dir = int(row / (c - a))
    else:
        row_dir = 1
    if d != b:
        col_dir = int(col / (d - b))
    else:
        col_dir = 1
    if row <= col:
        eql_move = row
         #we dont need to move any more rows now
    else:
        eql_move = col
    row = row - eql_move
    col = col - eql_move
    for i in range(1, eql_move + 1):  # adds (2,2),(3,3),...
        a = a + row_dir
        b = b + col_dir
        shortest_path.append(tuple([a, b]))

    for j in range(0, col):
        b = b + col_dir
        shortest_path.append(tuple([a, b]))
    for j in range(0, row):
        a = a + row_dir
        shortest_path.append(tuple([a, b]))

    length = len(shortest_path) - 1
    return length


# first create src and dest array where request values are greater than 0
src = []
dest = []
reqs = []
for i in range(0, len(src_arr)):
    if reqs_arr[i] > 0:
        src.append(src_arr[i])
        dest.append(dest_arr[i])
        reqs.append(reqs_arr[i])

capacity = 2
alpha = 1.5

paired_src = []
paired_dest = []
paired_req = []

for i in range(0, len(src)):
    if reqs[i] > capacity:
        for a in range(0, math.floor(reqs[i] / capacity)):
            paired_src.append(src[i])
            paired_dest.append(dest[i])
            paired_req.append(capacity)
        if (reqs[i] - (a + 1) * capacity) < capacity:  # (reqs[i]/capacity)<1 and (reqs[i]/capacity)>0:
            paired_src.append(src[i])
            paired_dest.append(dest[i])
            paired_req.append(reqs[i] - (a + 1) * capacity)
    else:
        paired_src.append(src[i])
        paired_dest.append(dest[i])
        paired_req.append(reqs[i])


paired_req[19]

sum(paired_req)

# calculating vehicle count
v_cou = 0
paired_src1 = []
paired_dest1 = []
paired_req1 = []
for i in range(0, len(paired_req)):
    if paired_req[i] == capacity:
        v_cou = v_cou + 1
    else:
        paired_src1.append(paired_src[i])
        paired_dest1.append(paired_dest[i])
        paired_req1.append(paired_req[i])



def ngb(src):
        k = 1
        coors = []
        src_x, src_y = map1[src]
        for i in range(-(k), k + 1):
            for j in range(-(k), k + 1):
                coors.append(map[(src_x + i, src_y + j)])
        for i in coors:
         if i in paired_src:
          return 1,i
check=[]
for i in range(0,len(paired_req)):
    if paired_src[i] not in check and (paired_req[i]==capacity or paired_req[i]==2.0):
        dest=paired_dest[i]
        in_lop=0
        for j in paired_src[i+1:]:
            if j==dest:
                in_lop=1
                v_cou=v_cou-1
                check.append(j)
                if paired_dest[paired_src.index(j)] in paired_src:
                  v_cou=v_cou-1
                  check.append(paired_src[paired_dest.index(paired_dest[paired_src.index(j)])])
                break
            if in_lop==0 and ngb(j)==1:
                b,a=ngb(j)
                v_cou=v_cou-1
                check.append(a)
                break


def check(b):

    for i in range(0, len(paired_req1)):

            dest = paired_dest1[i]
            for j in paired_dest1[paired_dest1.index(dest) + 1:]:
                if j in paired_src1:
                    v_cou = v_cou - 1
                    if paired_dest1[paired_src1.index(j)] in paired_src1:
                        v_cou = v_cou - 1
                elif ngb(j) == 1:
                    v_cou = v_cou - 1


# check if elements in paired vehicles can be paired

sum_pair = 0
paired_src12 = []
paired_dest12 = []
for i, j in zip(paired_src1, paired_dest1):

    det_i = len_sp(map1[i], map1[j])
    index = paired_dest1.index(j)
    pair = paired_req1[index]
    for a, b in zip(paired_src1[index + 1:], paired_dest1[index + 1:]):
        det_j = len_sp(map1[i], map1[a]) + len_sp(map1[a], map1[b]) + len_sp(map1[b], map1[j])
        if det_j >= det_i and ((a, b) not in paired_src12) and ((i, j) not in paired_src12):
            if det_i == 0 or (det_j / det_i) <= alpha:
                if (paired_req1[index] + paired_req1[paired_dest1.index(b)]) <= capacity:
                    pair = pair + paired_req1[paired_dest1.index(b)]
                    if pair < capacity:
                        paired_src12.append((a, b))
                    elif pair == capacity:
                        paired_src12.append((a, b))
                        paired_src12.append((i, j))
                        sum_pair = sum_pair + 1
                        check(b)
                        break
                    elif paired_dest1.index(j) == len(paired_dest1) - 1:
                        sum_pair = sum_pair + 1
                        check(b)
                else:
                    rem_cap = paired_req1[index] - capacity
                    pair = pair + rem_cap
                    paired_req1[index] = paired_req1[index] - rem_cap
        elif det_j < det_i and ((a, b) not in paired_src12) and ((i, j) not in paired_src12):
            if det_j == 0 or (det_i / det_j) <= alpha:
                if (paired_req1[index] + paired_req1[paired_dest1.index(b)]) <= capacity:
                    pair = paired_req1[index] + paired_req1[paired_dest1.index(b)]
                    if pair < capacity:
                        paired_src12.append((a, b))
                    elif pair == capacity:
                        paired_src12.append((a, b))
                        paired_src12.append((i, j))
                        sum_pair = sum_pair + 1
                        check(b)
                        break
                    elif paired_dest1.index(j) == len(paired_dest1) - 1:
                        sum_pair = sum_pair + 1
                        check(b)
                else:
                    rem_cap = paired_req1[index] - capacity
                    pair = pair + rem_cap
                    paired_req1[index] = paired_req1[index] - rem_cap

print("vehicles required",sum_pair+v_cou)
