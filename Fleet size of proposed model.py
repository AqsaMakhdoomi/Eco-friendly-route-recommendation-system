
import numpy as np
length_grid=577
rows_grid=24  #sqrt(length_grid-1)
cols_grid=24
grid=np.arange(1, length_grid, 1, dtype=int)
grid=grid.reshape((cols_grid, rows_grid))



res_G = np.load('../../', allow_pickle=True).tolist()  #run trainer file and put the value of res_G which contains predicted
#origin and destination of  requests
target_G = np.load('../../', allow_pickle=True).tolist() #run trainer file and put the value of target_G which contains actual
#origin and destination of  requests
res_D = np.load('../../', allow_pickle=True).tolist() #run trainer file and put the value of res_D which contains predicted
#origin of  requests
target_D = np.load('../../', allow_pickle=True).tolist() ##run trainer file and put the value of target_D which contains actual
#origin of  requests

res_G = np.asarray(res_G, dtype=np.float32)

res_G = res_G.reshape(576, 576)
# print(np.where(target_D==0))
hrs= [0, 41, 50, 10, 15, 29, 5, 6, 51, 13, 11, 26, 7, 12, 43, 19, 8, 23, 3, 1, 2, 4, 9, 55]


target_G = np.asarray(target_G, dtype=np.float32)
target_G = target_G.reshape(576, 576)

demand_before=np.sum(target_G)
from copy import deepcopy

target_G1 = deepcopy(target_G)


req = res_G


# Mapping
relative = []
for i in range(0, cols_grid):
    for j in range(0, rows_grid):
        relative.append(tuple([i, j]))
positional = list(range(0, rows_grid * cols_grid + 1, 1))
map = zip(relative, positional)
map = dict(map)  # maps (0,0) to 0
map1 = zip(positional, relative)  # maps 0 to (0,0)   i.e maps grid cells to coordinate
map1 = dict(map1)

src = (1, 2)
dest = (
    10, 11)
capacity = 2
alpha = 1.5

constant_veh=10
constant_veh1=10
# calculates shortest path
def SP(source, des):
    a, b = source
    c, d = des
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

    return shortest_path


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


shortest_path = SP(src, dest)

k = 2

a, b = src
c, d = dest
r_e = abs(c - a) + 1
c_e = abs(d - b) + 1
elem = r_e * c_e
wind_move = elem / (k * k)


def reverse(path):  # reverses list
    return path[::-1]



win_size = 4 * k + 1



i, j = src
ss = i - k
se = j - k
ds = i + k
de = j + k
A = range(ss, ds + 1)
B = range(se, de + 1)
loop = [tuple([a, b]) for a in A for b in B]
loop = np.array(loop)



def backslash(pathh):  # returns number of '/'
    b_s = 0
    for i in pathh:
        if i == '/':
            b_s = b_s + 1
    return b_s


def rep_elem(pathh):
    len_back = [0]
    cou = 0
    for i in pathh:
        if i != '/':
            cou = cou + 1
        else:
            len_back.append(cou + 1)
            cou = 0
    return len_back


def add_back(len_back):  # counts the length of len_back
    for i in range(1, len(len_back)):
        len_back[i] = len_back[i] + len_back[i - 1]
    return len_back


def rel(rel_src, rel_src1):
    ret = 0
    c, d = rel_src
    n1 = c - 1, d - 1
    n2 = c, d - 1
    n3 = c + 1, d - 1
    n4 = c - 1, d
    n5 = c, d
    n6 = c + 1, d
    n7 = c - 1, d + 1
    n8 = c, d + 1
    n9 = c + 1, d + 1

    if rel_src1 == n1 or rel_src1 == n2 or rel_src1 == n3 or rel_src1 == n4 or rel_src1 == n5 or rel_src1 == n6 or rel_src1 == n7 or rel_src1 == n8 or rel_src1 == n9:  # direct neighbors
        ret = 1
    return ret


def cal_num(path, start, end):
    start_index = path.index(start)
    end_index = path.index(end)
    return abs(end_index - start_index)



import numpy as np




sps = np.zeros((length_grid, length_grid))


def sps_cal():
    for i in range(0, length_grid - 1):
        for j in range(0, length_grid - 1):
            sps[i][j] = len_sp(map1[i], map1[j])
    return sps


sps_cal()

import math


def maxim(src, k, first, t, backtrack, rel_src1, an):  # calculates maximum from source within k neighbors
    tri, trj = src
    coord = []
    temp = []

    for xi in range(0, k + 1):  # forward neighbors
        for yj in range(0, k + 1):

            if window_k[tri + xi][trj + yj] >= 0 and xi + yj != 0 and window_k[tri + xi][trj + yj] != rel_map1[
                src]:
                if k == 1 and (
                        sps[rel_map1[rel_src1]][window_k[tri + xi][trj + yj]] + t <= alpha * sps[rel_map1[rel_src1]][
                    backtrack[0]]):
                    temp.append(req[window_k[tri + xi][trj + yj]][rel_map1[src]])
                    coord.append(window_k[tri + xi][trj + yj])
                if k != 1:
                    temp.append(req[rel_map1[src]][window_k[tri + xi][trj + yj]])
                    coord.append(window_k[tri + xi][trj + yj])

    for xi in range(1, k + 1):  # down backward neighbors
        for yj in range(0, k + 1):  # when we go left we subtract from 1 beacuse 0th col is added
            if window_k[tri - xi][trj + yj] >= 0 and xi + yj != 0 and window_k[tri - xi][trj + yj] != rel_map1[
                src]:

                if k == 1 and (
                        sps[rel_map1[rel_src1]][window_k[tri - xi][trj + yj]] + t <= alpha * sps[rel_map1[rel_src1]][
                    backtrack[0]]):
                    temp.append(req[window_k[tri - xi][trj + yj]][rel_map1[src]])
                    # coord.append((i-xi,j+yj))
                    coord.append(window_k[tri - xi][trj + yj])
                if k != 1:  # and (window_k[i-xi][j+yj] not in path):
                    temp.append(req[rel_map1[src]][window_k[tri - xi][trj + yj]])
                    coord.append(window_k[tri - xi][trj + yj])

    for xi in range(0, k + 1):  # up right neighbors
        for yj in range(1, k + 1):
            if window_k[tri + xi][trj - yj] >= 0 and xi + yj != 0 and window_k[tri + xi][trj - yj] != rel_map1[
                src]:
                if k == 1 and (
                        sps[rel_map1[rel_src1]][window_k[tri + xi][trj - yj]] + t <= alpha * sps[rel_map1[rel_src1]][
                    backtrack[0]]):
                    temp.append(req[window_k[tri + xi][trj - yj]][rel_map1[src]])
                    coord.append(window_k[tri + xi][trj - yj])
                if k != 1:
                    temp.append(req[rel_map1[src]][window_k[tri + xi][trj - yj]])
                    coord.append(window_k[tri + xi][trj - yj])

    for xi in range(1, k + 1):  # up neighbors
        for yj in range(1, k + 1):
            if window_k[tri - xi][trj - yj] >= 0 and xi + yj != 0 and window_k[tri - xi][trj - yj] != rel_map1[
                src]:
                if k == 1 and (
                        sps[rel_map1[rel_src1]][window_k[tri - xi][trj - yj]] + t <= alpha * sps[rel_map1[rel_src1]][
                    backtrack[0]]):
                    temp.append(req[window_k[tri - xi][trj - yj]][rel_map1[src]])
                    coord.append(window_k[tri - xi][trj - yj])
                if k != 1:  # and (window_k[i-xi][j-yj] not in path):
                    temp.append(req[rel_map1[src]][window_k[tri - xi][trj - yj]])
                    coord.append(window_k[tri - xi][trj - yj])


    grids_traversed = len(coord)

    max_value = max(temp)
    max_index = temp.index(max_value)
    pos1 = coord[max_index]  # would be grid index
    if sum(temp) == 0 and k != 1:

        p = []
        if first == 1:
            p = SP(map1[rel_map1[src]],
                   dest)  # dest is final dest, src is point in window from which we want to reach dest

        else:
            p = SP(map1[rel_map1[src]], map1[rel_map1[rel_src1]])
        q = []
        loop_g = 0
        for s in p:
            q.append(map[s])

        for pos in reverse(q):
            if pos in window_k and q.index(pos) == q.index(rel_map1[src]) + 1:
                pos1 = pos
                loop_g = 1
            if loop_g == 0:
                pos1 = coord[random.randint(0, len(temp) - 1)]

    elif sum(temp) == 0 and k == 1:

        p = []
        if first == 1:
            p = SP(map1[backtrack[0]],
                   map1[rel_map1[src]])  # dest is final dest, src is point in window from which we want to reach dest

        else:
            p = SP(map1[rel_map1[src]], map1[rel_map1[rel_src1]])

        q = []
        loop_g = 0
        for s in p:
            q.append(map[s])

        for pos in reverse(q):

            if pos in window_k and q.index(pos) == q.index(rel_map1[src]) + 1:
                pos1 = pos
                loop_g = 1
            if loop_g == 0:
                pos1 = coord[random.randint(0, len(temp) - 1)]

    an.append(tuple([rel_map1[src], pos1]))
    arr_str1.append(tuple([rel_map1[src], pos1]))
    arr_str2.append(abs(target_G[rel_map1[src]][pos1]))
    req[rel_map1[src]][pos1] = req[rel_map1[src]][pos1] - capacity  # capacity is dynamic capacity
    target_G[rel_map1[src]][pos1] = max(0, target_G[rel_map1[src]][
        pos1] - capacity)
    pos = rel_map[pos1]

    backtrack.append(pos1)
    prev = rel_map1[src]
    rel_src = pos
    t = t + 1
    return backtrack, rel_src, t, arr_str1, arr_str2, an


path = [map[src]]
path_backup = []
pat2_new = [map[src]]
p_b = [map[src]]
capacity = 2
capacity1 = capacity
gc = 0
map_req = []
arr_str1 = []
arr_str2 = []


def cal_max(src):  # calculates path within all windows
    zr = 0
    gc = 1
    firs = 0
    t = 0
    an = []
    backtrack = []  # empty backtrack first
    if firs == 0:
        rel_src = rel_map[map[src]]
    else:
        rel_src = src
        firs = 1
    rel_src1 = rel_src
    capacity =
    capacity1 = capacity

    backtrack, rel_src, t, arr_str1, arr_str2, an = maxim(rel_src, k, 1, t, backtrack, rel_src1, an)  # first time k=2
    map_req = zip(arr_str1, arr_str2)

    sp = SP(src, map1[backtrack[0]])  # SP in window
    l_sp = len(sp)  # length of shortest path in window
    detour = 1.7
    threshold = math.ceil(detour * l_sp)  # distance we can travel
    stop = rel(rel_src, rel_src1)
    if stop != 1:
        gc = gc + 1
    # print("gc is",gc)
    if stop == 1:  # move window forward
        # add bactrack elements in reverse to path(which contains src only) and pass rel_src to it and make it src of new window
        reversed_backtrack = reverse(backtrack)
        for i in reversed_backtrack:
            path.append(i)
            p_b.append(i)
        capacity = capacity1  # after window is moved capacity becomes full again
        path.append('/')
        p_b.append('/')
        i_l = 0
        co = 0
        path2_new = []
        path1 = path
        path2 = reverse(path1[:-1])
        while path2[i_l] != '/' and co != (len(path2) - 1):
            path2_new.append(path2[i_l])
            i_l = i_l + 1
            co = co + 1

        path2_new = reverse(path2_new)

        for el_y in path2_new:
            ind = len(path1) - 1 - path1[::-1].index(el_y)
            if el_y != '/':

                for d in path1[:-(len(path1) - ind)]:
                    if d != '/':
                        if d != el_y and (d, el_y) not in an:
                            arr_str1.append(tuple([d, el_y]))
                            arr_str2.append(abs(target_G[d][el_y]))
                            map_req = zip(arr_str1, arr_str2)
                        else:
                            indx = path1.index(d)
                            path1.remove(d)
                            exit_cur = 1
                            break


        return backtrack, rel_src, gc, map_req, an

    else:
        while True:  # rel_src is not immediate neighbor
            backtrack, rel_src, t, arr_str1, arr_str2, an = maxim(rel_src, 1, 1, t, backtrack, rel_src1, an)  # then k=1
            map_req = zip(arr_str1, arr_str2)

            reversed_backtrack = reverse(backtrack)  # not required

            if stop != 1:
                gc = gc + 1
            stop = rel(rel_src, rel_src1)
            if stop == 1:
                reversed_backtrack = reverse(backtrack)
                for i in reversed_backtrack:
                    path.append(i)
                    p_b.append(i)
                path.append('/')
                p_b.append('/')
                path1 = path
                i_l = 0

                co = 0
                path2 = reverse(path1[:-1])
                path2_new = []
                while path2[i_l] != '/' and co != (len(path2) - 1):
                    path2_new.append(path2[i_l])
                    i_l = i_l + 1
                    co = co + 1

                path2_new = reverse(path2_new)

                for el_y in path2_new:
                    ind = len(path1) - 1 - path1[::-1].index(el_y)
                    if el_y != '/':

                        for d in path1[:-(len(path1) - ind)]:
                            if d != '/':
                                if d != el_y and (d, el_y) not in an:
                                    arr_str1.append(tuple([d, el_y]))
                                    arr_str2.append(abs(target_G[d][el_y]))
                                    map_req = zip(arr_str1, arr_str2)
                                else:
                                    indx = path1.index(d)
                                    path1.remove(d)
                                    exit_cur = 1
                                    break


                return backtrack, rel_src, gc, map_req, an


def ter(dest, window_k, path):
    if map.get(dest) in path:
        return 1
    for j in range(0, 4 * k + 1):
        for i in window_k[j]:

            if map.get(dest) == i:


                return 1
    return 0


import math


def mn(mid):
    shor_rev = shortest_path[::-1]

    r, t = mid
    dist = []
    cor = []
    for i in shor_rev:
        if map.get(i) in window_k:
            l, b = rel_map[map[i]]
            dist.append(tuple([math.sqrt((l - r) ** 2 + (b - t) ** 2)]))
            cor.append(rel_map[map[i]])

    if not dist:
        m = k - 1
        n = k - 1
    else:
        kc = min(dist)
        index = dist.index(kc)
        val = cor[index]
        c, v = val
        m = c - r

        n = v - t

    return m, n


rel_map = {}
rel_map1 = {}


def createwindow(m, n, sq):  # dest_k is rel_src  sr is a(last point where shortest path touches window)

    v, w = sq
    sr_ = v, w


    window_k[middle][middle] = map[sr_]
    o, u = sr_

    arr = []
    arr2 = []

    if m >= 0 and n <= 0:  # we can go down normally and up k-m+1 times
        # we can go left normally and right k+n+1 times
        for i in range(0,
                       k + 1):  # as m=1 which means it is 1 col above shortest path=> we can move k-m points above only
            for kx in range(0, k - m + 1):  # down
                for j in range(0, k + n + 1):  # right
                    for jz in range(0, k + 1):  # up
                        # if map[(o+kx,u-j)]

                        if map.get((o + i, u - jz)) != None:
                            window_k[middle + i][middle - jz] = map[(o + i, u - jz)]
                            ##########
                            arr.append(tuple([middle + i, middle - jz]))
                            arr2.append(map[(o + i, u - jz)])

                        if map.get((o - kx, u - jz)) != None:
                            window_k[middle - kx][middle - jz] = map[(o - kx, u - jz)]
                            ###########
                            arr.append(tuple([middle - kx, middle - jz]))
                            arr2.append(map[(o - kx, u - jz)])

                        if map.get((o + i, u + j)) != None:
                            window_k[middle + i][middle + j] = map[(o + i, u + j)]
                            ##########
                            arr.append(tuple([middle + i, middle + j]))
                            arr2.append(map[(o + i, u + j)])

                        if map.get((o - kx, u + j)) != None:
                            window_k[middle - kx][middle + j] = map[(o - kx, u + j)]
                            #    print("tuple is",tuple([middle+kx,middle-j]))
                            #########

                            arr.append(tuple([middle - kx, middle + j]))

                            arr2.append(map[(o - kx, u + j)])

    elif m >= 0 and n >= 0:
        # we can go down normally and up k-m+1 times
        # we can go right normally and left k-n+1 times
        for i in range(0,
                       k + 1):  # as m=1 which means it is 1 col above shortest path=> we can move k-m points above only
            for j in range(0, k - n + 1):
                for kx in range(0, k - m + 1):
                    for jz in range(0, k + 1):
                        if map.get((o + i, u - j)) != None:
                            window_k[middle + i][middle - j] = map[(o + i, u - j)]
                            arr.append(tuple([middle + i, middle - j]))
                            arr2.append(map[(o + i, u - j)])

                        if map.get((o - kx, u - j)) != None:
                            window_k[middle - kx][middle - j] = map[(o - kx, u - j)]
                            arr.append(tuple([middle - kx, middle - j]))
                            arr2.append(map[(o - kx, u - j)])

                        if map.get((o + i, u + jz)) != None:
                            window_k[middle + i][middle + jz] = map[(o + i, u + jz)]
                            arr.append(tuple([middle + i, middle + jz]))
                            arr2.append(map[(o + i, u + jz)])

                        if map.get((o - kx, u + jz)) != None:
                            window_k[middle - kx][middle + jz] = map[(o - kx, u + jz)]
                            #     print("tuple is",tuple([middle+i,middle-j]))
                            arr.append(tuple([middle - kx, middle + jz]))
                            arr2.append(map[(o - kx, u + jz)])




    ###
    elif m <= 0 and n <= 0:
        # we can go up normally and down k+m+1
        # we can go left normally and right k+n+1 times

        for i in range(0,
                       k + m + 1):  # as m=1 which means it is 1 col above shortest path=> we can move k-m points above only
            for j in range(0, k + n + 1):
                for kx in range(0, k + 1):
                    for jz in range(0, k + 1):

                        if map.get((o + i, u - jz)) != None:
                            window_k[middle + i][middle - jz] = map[(o + i, u - jz)]
                            arr.append(tuple([middle + i, middle - jz]))
                            arr2.append(map[(o + i, u - jz)])

                        if map.get((o - kx, u - jz)) != None:
                            window_k[middle - kx][middle - jz] = map[(o - kx, u - jz)]
                            arr.append(tuple([middle - kx, middle - jz]))
                            arr2.append(map[(o - kx, u - jz)])

                        if map.get((o + i, u + j)) != None:
                            window_k[middle + i][middle + j] = map[(o + i, u + j)]
                            arr.append(tuple([middle + i, middle + j]))
                            arr2.append(map[(o + i, u + j)])

                        if map.get((o - kx, u + j)) != None:
                            window_k[middle - kx][middle + j] = map[(o - kx, u + j)]

                            arr.append(tuple([middle - kx, middle + j]))
                            arr2.append(map[(o - kx, u + j)])





    elif m <= 0 and n >= 0:

        for i in range(0,
                       k + m + 1):  # as m=1 which means it is 1 col above shortest path=> we can move k-m points above only
            for j in range(0, k + 1):
                for kx in range(0, k + 1):
                    for jz in range(0, k - n + 1):

                        if map.get((o + i, u - jz)) != None:
                            window_k[middle + i][middle - jz] = map[(o + i, u - jz)]
                            arr.append(tuple([middle + i, middle - jz]))
                            arr2.append(map[(o + i, u - jz)])

                        if map.get((o - kx, u - jz)) != None:
                            window_k[middle - kx][middle - jz] = map[(o - kx, u - jz)]
                            arr.append(tuple([middle - kx, middle - jz]))
                            arr2.append(map[(o - kx, u - jz)])

                        if map.get((o + i, u + j)) != None:
                            window_k[middle + i][middle + j] = map[(o + i, u + j)]
                            arr.append(tuple([middle + i, middle + j]))
                            arr2.append(map[(o + i, u + j)])

                        if map.get((o - kx, u + j)) != None:
                            window_k[middle - kx][middle + j] = map[(o - kx, u + j)]

                            arr.append(tuple([middle - kx, middle + j]))

                            arr2.append(map[(o - kx, u + j)])

    rel_map = zip(arr2, arr)
    rel_map = dict(rel_map)  # contains src(not relative src) and its original mapping through map[]

    rel_map1 = zip(arr, arr2)
    rel_map1 = dict(rel_map1)


    return window_k, rel_map, rel_map1


arr_str1 = []
arr_str2 = []


def sum_tg(target_G):
    sm = 0
    for i in range(0, len(target_G)):
        for j in range(0, len(target_G)):
            sm = sm + target_G[i][j]
    return sm




import random

const = 3

infinity_check = 0
veh_cou = 0

import numpy as np
stored_value=0
while sum_tg(target_G1) != 0:
    path = []
    path1 = []
    inifinity_check = infinity_check + 1
    lp = 0
    cal_m = 0
    first = 0
    ap = np.argsort(target_G[20])
    srcx, srcy = src
    if (srcx + const - 1) <= 78 and (srcy + const - 1) <= 67:
        dest = srcx + const - 1, srcy + const - 1
    else:
        dest = (random.randint(0, srcx), random.randint(0, srcy))

    shortest_path = SP(src, dest)

    first = 1

    l = 0
    ret = 0

    first = 0
    lp = 0
    const = 5
    while ret != 1:

        if first == 0:
            m = 0
            n = 0

        else:
            m, n = mn(rel_map[backtrack[0]])

        window_k = np.arange(-win_size * win_size, 0, dtype=int)

        window_k = window_k.reshape(win_size, win_size)
        middle = math.floor(len(window_k) / 2)
        window_k, rel_map, rel_map1 = createwindow(m, n, src)

        backtrack, rel_src, grids_traversed, map_req, an = cal_max(
            src)
        cal_m = cal_m + 1
        src = map1[backtrack[0]]  # updating src here

        first = 1


        tr = 0
        pathh = path[:]

        ret = ter(shortest_path[-1], window_k, path)

        if ret == 1:

            rel_src = rel_map[map[dest]]
            rel_src1 = rel_map[backtrack[0]]
            backtrack = []
            backtrack.append(map[dest])
            exit = 0
            t = 1

            while exit != 1:

                stop = rel(rel_src, rel_src1)

                tr = tr + 1
                if stop == 1:
                    exit = 1
                    reversed_backtrack = reverse(backtrack)
                    for i in reversed_backtrack:
                        path.append(i)
                        p_b.append(i)
                    path.append('/')
                    p_b.append('/')

                    path1 = path

                    i_l = 0
                    path2 = reverse(path1[:-1])
                    path2_new = []
                    co = 0
                    while path2[i_l] != '/' and co != (len(path2) - 1):
                        path2_new.append(path2[i_l])
                        i_l = i_l + 1
                        co = co + 1

                    path2_new = reverse(path2_new)

                    for el_y in path2_new:

                        ind = len(path1) - 1 - path1[::-1].index(el_y)

                        if el_y != '/':


                            for d in path1[:-(len(path1) - ind)]:
                                if d != '/':
                                    if d != el_y and (d, el_y) not in an:
                                        arr_str1.append(tuple([d, el_y]))
                                        arr_str2.append(abs(target_G[d][el_y]))
                                        map_req = zip(arr_str1, arr_str2)
                                    else:

                                        indx = path1.index(d)
                                        path1.remove(d)
                                        exit_cur = 1
                                        break

                else:
                    backtrack, rel_src, t, arr_str1, arr_str2, an = maxim(rel_src, 1, 0, t, backtrack, rel_src1,
                                                                          an)

                    map_req = zip(arr_str1, arr_str2)



    path1 = []
    for i in path:
        if i != '/':
            path1.append(i)
    count = 6

    path = path1[:count]
    path = [*set(path)]
    s = []
    d = []
    l = []
    for i in path:
        for j in path[path.index(i):]:
            s.append(i)
            d.append(j)
            l.append(path.index(j) - path.index(i))

    sort_index = np.argsort(np.array(l))
    indices = reverse(list(sort_index))
    sorted_s = []
    sorted_d = []
    for i in indices:
        sorted_s.append(s[i])
        sorted_d.append(d[i])
    capacity = 2
    cap_covered = 0
    cap_left = capacity
    veh_cou = veh_cou + 1

    stored_value=stored_value+count-len(path)

    if (stored_value>=const):
        constant_veh = constant_veh + 1
        stored_value = stored_value-const


    for i, j in zip(sorted_s, sorted_d):


        if target_G1[i][j]:

            cap_covered = cap_covered + min(cap_left, target_G1[i][j])
            target_G1[i][j] = target_G1[i][j] - min(cap_left, target_G1[i][j])
            res_G[i][j] = res_G[i][j] - min(cap_left,
                                            res_G[i][j])
            cap_left=capacity-cap_covered

            if cap_covered == capacity:
                break

    src = (random.randint(0, 14), random.randint(0, 14))


rows = len(np.where(target_G1 > 0))
cols = len(np.where(target_G1 > 0)[0])
length1 = rows * cols
print("vehicles required",demand_before-np.sum(target_G1))
print("vehicles required", math.ceil((demand_before*constant_veh1)/(demand_before-np.sum(target_G1))))

