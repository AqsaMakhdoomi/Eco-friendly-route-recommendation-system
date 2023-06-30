#This is the code for Backward greedy algorithm
import numpy as np

length_grid = 362
rows_grid = 19  # sqrt(length_grid-1)
cols_grid = 19
grid = np.arange(1, length_grid, 1, dtype=int)
grid = grid.reshape((cols_grid, rows_grid))


res_G = np.load('../../', allow_pickle=True).tolist()  #run trainer file and put the value of res_G which contains predicted
#origin and destination of  requests
target_G = np.load('../../', allow_pickle=True).tolist() #run trainer file and put the value of target_G which contains actual
#origin and destination of  requests
res_D = np.load('../../', allow_pickle=True).tolist() #run trainer file and put the value of res_D which contains predicted
#origin of  requests
target_D = np.load('../../', allow_pickle=True).tolist() ##run trainer file and put the value of target_D which contains actual
#origin of  requests

res_G = np.asarray(res_G, dtype=np.float32)
res_G = res_G.reshape(361, 361)


target_G = np.asarray(target_G, dtype=np.float32)
target_G = target_G.reshape(361, 361)

req = res_G

# Mapping
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

src = (6, 9)
dest = (14, 13)
capacity = 2


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
    # find min(row,col)
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

k = 3

a, b = src
c, d = dest


def reverse(path):  # reverses list
    return path[::-1]


import math

win_size = 4 * k + 1


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




def efficiency(pathh, len_back, b_s,
               map_req):  # b_s is num of backslashes and  len_back gives an array which contains the length of path in each window
    print("map_req in efficiency", map_req)
    orders = 0
    arr_a = []
    ok = 0
    el = 0
    added_elem = []
    efficiency = 0
    lk = []
    ct = 0
    capacity = 2
    c = capacity
    arr_a
    j = 0
    cz = 0
    vis1 = np.zeros(len(map_req))  # -1
    vis2 = np.zeros(len(map_req))  # -1
    pass_g = 0  # passengers per grid
    copied_path = np.zeros(len(pathh) - b_s)
    copied_path = list(copied_path)
    sum_eff = 0
    for i in range(len(pathh)):
        if pathh[i] != '/':
            copied_path[j] = pathh[i]
            j = j + 1
    l_c = len(copied_path)
    i_c = 1
    p_k = copied_path
    reversed_list = reverse(copied_path)
    count = 0
    for iz in p_k:
        arr_a = []
        for kl in range(i_c - 2, -1, -1):
            if p_k[i_c - 1] != p_k[kl]:
                for h in range(len(map_req) - 1):
                    if map_req[h][0] == (p_k[kl], iz) and vis1[map_req.index((map_req[h][0], map_req[h][1]))] == 0:
                        cz = map_req[h][1]

                        if cz and [p_k[kl], iz] in added_elem:
                            capacity = min(c, capacity + cz)
                            ind1 = map_req.index((map_req[h][0], cz))
                            vis1[ind1] = 1
            else:
                kl = -1
                poly = 1
        i_c = i_c + 1
        count = count + 1

        lk = reversed_list[:-count]
        if iz in lk:
            ind_l = lk.index(iz)
            rep = 1
        else:
            ind_l = len(lk) - 1
            rep = 0

        if rep == 0:

            for a in lk[0:ind_l + 1]:
                if a not in arr_a:
                    arr_a.append(a)

                    if capacity > 0:

                        val = cal_num(copied_path, a, iz)
                        if val <= alpha * len_sp(map1[iz], map1[a]):
                            for h in range(len(map_req) - 1):
                                if map_req[h][0] == (iz, a):
                                    cz = map_req[h][1]
                            capacity = max(0, capacity - cz)

                            added_elem.append([iz, a])

        else:
            for a in lk[ind_l + 1:len(lk)]:
                if a not in arr_a:
                    arr_a.append(a)

                    if capacity > 0:
                        val = cal_num(copied_path, a, iz)
                        if val <= alpha * len_sp(map1[iz], map1[a]):
                            for h in range(len(map_req) - 1):
                                if map_req[h][0] == (iz, a) and vis2[
                                    map_req.index((map_req[h][0], map_req[h][1]))] == 0:
                                    cz = map_req[h][1]
                                    capacity = max(0, capacity - cz)
                                    ind2 = map_req.index((map_req[h][0], cz))
                                    vis2[ind2] = 1
                            added_elem.append([iz, a])

        pass_g = pass_g + c - capacity
        if c - 2 >= capacity:  # if there are atleast 2 passengers in the vehicle then  ridesharing is successful
            ok = ok + 1
        if c - 1 == capacity:
            orders = orders + 1
        efficiency = (c - capacity) / c
        sum_eff = sum_eff + efficiency
    pass_g = pass_g / l_c
    sum_eff = sum_eff / l_c
    print("sum_eff is", sum_eff)
    perc_ride_without = (orders * 100) / l_c
    perc_ride_with = (ok * 100) / l_c

    print("percentage of orders without ridesharing are", perc_ride_without)
    print("percentage of orders with ridesharing are", perc_ride_with)

    print("passengers per grid", pass_g)
    return added_elem


sps = np.zeros((length_grid, length_grid))


def sps_cal():
    for i in range(0, length_grid - 1):
        for j in range(0, length_grid - 1):
            sps[i][j] = len_sp(map1[i], map1[j])
    return sps


sps_cal()


def cal_num(path, start, end):
    start_index = path.index(start)
    end_index = path.index(end)
    return abs(end_index - start_index)


def comm_dist(path, e1, e2):
    for i in range(0, len(path)):
        if path[i] == e1[0][0]:
            ind_i = i
    for j in range(0, len(path)):
        if path[j] == e1[0][1]:
            ind_j = j
    path_1 = path[ind_i:ind_j + 1]

    for i in range(0, len(path)):
        if path[i] == e2[0][0]:
            ind2_i = i
    for j in range(0, len(path)):
        if path[j] == e2[0][1]:
            ind2_j = j
    path_2 = path[ind2_i:ind2_j + 1]

    common_el = set(path_1).intersection(path_2)
    return len(common_el) - 1



def n_cars(path, lenb, b_s):  # numnber of cars used for path with and without ridesharing
    n_without = 0
    count = 1
    copied_path = np.zeros(lenb - b_s)
    copied_path = list(copied_path)
    j = 0
    for i in range(len(path)):
        if path[i] != '/':
            copied_path[j] = path[i]
            j = j + 1

    for i in copied_path:
        copied_pat = copied_path[count:]
        for j in copied_pat:
            n_without = n_without + target_G[i][j]
        count = count + 1
    print("Number of cars without ridesharing", n_without)

    n_with = n_without / c  # c is capacity
    print("Number of cars with ridesharing", n_with)
    return n_without


# SHORTEST PATH FULL WITHOUT k
def eff_shor(pathh):
    orders = 0
    ok = 0
    el = 0
    added_elem = []
    efficiency = 0
    lk = []
    ct = 0
    capacity = 2
    c = capacity
    j = 0
    pass_g = 0  # passengers per grid
    copied_path = np.zeros(len(pathh))
    copied_path = list(copied_path)

    sum_eff = 0
    for i in range(len(pathh)):
        if pathh[i] != '/':
            copied_path[j] = pathh[i]
            j = j + 1
    print("copied path", copied_path)
    l_c = len(copied_path)
    i_c = 1
    p_k = copied_path
    reversed_list = reverse(copied_path)
    count = 0
    for iz in p_k:
        for kl in range(i_c - 2, -1, -1):
            if target_G[p_k[kl]][iz] and [p_k[kl], iz] in added_elem:
                capacity = min(c, capacity + target_G[p_k[kl]][iz])
        i_c = i_c + 1
        count = count + 1
        lk = reversed_list[:-count]
        for a in lk:
            if capacity > 0:
                val = cal_num(copied_path, a, iz)
                if val <= alpha * len_sp(map1[iz], map1[a]):
                    capacity = max(0, capacity - target_G[iz][a])
                    added_elem.append([iz, a])

        pass_g = pass_g + c - capacity

        if c - 2 >= capacity:

            ok = ok + 1
        if c - 1 == capacity:
            orders = orders + 1


        efficiency = (c - capacity) / c
        sum_eff = sum_eff + efficiency
    pass_g = pass_g / l_c
    sum_eff = sum_eff / l_c
    print("sum_eff is", sum_eff)
    perc_ride_without = (orders * 100) / l_c
    perc_ride_with = (ok * 100) / l_c

    print("percentage of orders without ridesharing are", perc_ride_without)
    print("percentage of orders with ridesharing are", perc_ride_with)

    print("passengers per grid", pass_g)


alpha = 1.5

pos1 = -1
l = k
grids_traversed = 0
mat = np.zeros([len(req), len(req)], dtype=int)
import random


def maxim(src, k, first, t, backtrack, rel_src1, an):  # calculates maximum from source within k neighbors
    l = k
    tri, trj = src
    coord = []
    temp = []
    for xi in range(0, k + 1):  # forward neighbors
        for yj in range(0, k + 1):
            if window_k[tri + xi][trj + yj] >= 0 and xi + yj != 0 and window_k[tri + xi][trj + yj] != rel_map1[
                src]:
                if k == 1 and (sps[window_k[tri + xi][trj + yj]][backtrack[0]] + t <= alpha * sps[rel_map1[rel_src1]][
                    backtrack[0]]):
                    temp.append(req[window_k[tri + xi][trj + yj]][backtrack[0]])
                    coord.append(window_k[tri + xi][trj + yj])

                if k != 1:
                    temp.append(req[rel_map1[src]][window_k[tri + xi][trj + yj]])
                    coord.append(window_k[tri + xi][trj + yj])

    for xi in range(1, k + 1):  # down backward neighbors
        for yj in range(0, k + 1):  # when we go left we start from 1 beacuse 0th col is added
            #  print(window_k[i-xi][j+yj])
            if window_k[tri - xi][trj + yj] >= 0 and xi + yj != 0 and window_k[tri - xi][trj + yj] != rel_map1[
                src]:

                if k == 1 and (sps[window_k[tri - xi][trj + yj]][backtrack[0]] + t <= alpha * sps[rel_map1[rel_src1]][
                    backtrack[0]]):
                    temp.append(req[window_k[tri - xi][trj + yj]][backtrack[0]])
                    coord.append(window_k[tri - xi][trj + yj])
                if k != 1:
                    temp.append(req[rel_map1[src]][window_k[tri - xi][trj + yj]])
                    coord.append(window_k[tri - xi][trj + yj])

    for xi in range(0, k + 1):  # up right neighbors
        for yj in range(1, k + 1):
            if window_k[tri + xi][trj - yj] >= 0 and xi + yj != 0 and window_k[tri + xi][trj - yj] != rel_map1[
                src]:
                if k == 1 and (sps[window_k[tri + xi][trj - yj]][backtrack[0]] + t <= alpha * sps[rel_map1[rel_src1]][
                    backtrack[0]]):
                    temp.append(req[window_k[tri + xi][trj - yj]][backtrack[0]])
                    coord.append(window_k[tri + xi][trj - yj])
                if k != 1:
                    temp.append(req[rel_map1[src]][window_k[tri + xi][trj - yj]])
                    coord.append(window_k[tri + xi][trj - yj])

    for xi in range(1, k + 1):  # up neighbors
        for yj in range(1, k + 1):
            if window_k[tri - xi][trj - yj] >= 0 and xi + yj != 0 and window_k[tri - xi][trj - yj] != rel_map1[
                src]:
                if k == 1 and (sps[window_k[tri - xi][trj - yj]][backtrack[0]] + t <= alpha * sps[rel_map1[rel_src1]][
                    backtrack[0]]):

                    temp.append(req[window_k[tri - xi][trj - yj]][backtrack[0]])
                    coord.append(window_k[tri - xi][trj - yj])
                if k != 1:
                    temp.append(req[rel_map1[src]][window_k[tri - xi][trj - yj]])
                    coord.append(window_k[tri - xi][trj - yj])


    grids_traversed = len(coord)
    if not temp:
        if window_k[middle - 1][middle] > 0:
            pos1 = window_k[middle - 1][middle]

        elif window_k[middle][middle - 1] > 0:
            pos1 = window_k[middle][middle - 1]

        elif window_k[middle + 1][middle] > 0:
            pos1 = window_k[middle + 1][middle]
        elif window_k[middle][middle + 1] > 0:
            pos1 = window_k[middle][middle + 1]

    else:
        max_value = max(temp)
        max_index = temp.index(max_value)

        pos1 = coord[max_index]
        pos = rel_map[pos1]

    prev = rel_map1[src]
    rel_src = pos
    t = t + 1
    if sum(temp) == 0 and k != 1:

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


    backtrack.append(pos1)
    an.append(tuple([rel_map1[src], pos1]))
    arr_str1.append(tuple([rel_map1[src], pos1]))
    arr_str2.append(abs(target_G[rel_map1[src]][pos1]))
    req[rel_map1[src]][pos1] = req[rel_map1[src]][pos1] - capacity  # capacity is dynamic capacity
    target_G[rel_map1[src]][pos1] = max(0, target_G[rel_map1[src]][pos1] - capacity)  # capacity is dynamic capacity

    if k != 1:
        return backtrack, src, t, arr_str1, arr_str2, an
    else:
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


def cal_max(src):  # calculates full path within all windows
    gc = 1
    firs = 0
    t = 0
    an = []
    backtrack = []
    if firs == 0:
        rel_src = rel_map[map[src]]
    else:
        rel_src = src
        firs = 1
    rel_src1 = rel_src
    capacity = 2
    capacity1 = capacity

    backtrack, rel_src, t, arr_str1, arr_str2, an = maxim(rel_src, k, 1, t, backtrack, rel_src1,
                                                          an)
    map_req = zip(arr_str1, arr_str2)
    sp = SP(src, map1[backtrack[0]])
    l_sp = len(sp)

    detour = 2
    threshold = math.ceil(detour * l_sp)  # distance we can travel
    stop = rel(rel_src, rel_map[backtrack[0]])
    if stop != 1:
        gc = gc + 1

    if stop == 1:  # move window forward
        # add bactrack elements in reverse to path(which contains src only) and pass rel_src to it and make it src of new window
        reversed_backtrack = reverse(backtrack)
        xc = backtrack[0]
        backtrack1 = backtrack[1:]
        for i in backtrack1:
            path.append(i)
            p_b.append(i)
        path.append(xc)
        path.append('/')
        p_b.append(xc)
        p_b.append('/')
        capacity = capacity1
        i_l = 0
        co = 0
        path2_new = []
        path1 = path
        indx = -1
        path2 = reverse(path1[:-1])
        while path2[i_l] != '/' and co != (len(path2) - 1):
            path2_new.append(path2[i_l])
            i_l = i_l + 1
            co = co + 1

        path2_new = reverse(path2_new)

        for el_y in path2_new:

            ind = len(path1) - 1 - path1[::-1].index(el_y)
            if el_y != '/':

                for d in reverse(path1[0:-(len(path1) - ind)]):
                    if d != '/' and (d, el_y) not in an:
                        if d != el_y:
                            arr_str1.append(tuple([d, el_y]))
                            arr_str2.append(abs(target_G[d][el_y]))
                            map_req = zip(arr_str1, arr_str2)
                        else:

                            path1.remove(d)

                            break
        try:
            map_req
        except NameError:
            map_req = None

        if map_req is None:
            map_req = []
        #  print("empty")
        else:
            looo = 1
        return backtrack, rel_src, gc, map_req, an

    else:
        while True:
            reversed_backtrack = reverse(backtrack)

            backtrack, rel_src, t, arr_str1, arr_str2, an = maxim(rel_src, 1, 1, t, backtrack, rel_src1, an)  # then k=1
            reversed_backtrack = reverse(backtrack)  # not required
            # det_track=det_track+1
            map_req = zip(arr_str1, arr_str2)
            if stop != 1:
                gc = gc + 1
            stop = rel(rel_src, rel_map[backtrack[0]])
            if stop == 1:
                cx = backtrack[0]
                backtrack1 = backtrack[1:]
                for i in backtrack1:
                    path.append(i)
                    p_b.append(i)
                path.append(cx)
                path.append('/')
                p_b.append(cx)
                p_b.append('/')
                path1 = path
                i_l = 0
                indx = -1
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
                        for d in reverse(path1[0:-(len(path1) - ind)]):
                            if d != '/' and (d, el_y) not in an:
                                if d != el_y:
                                    arr_str1.append(tuple([d, el_y]))
                                    arr_str2.append(abs(target_G[d][el_y]))
                                    map_req = zip(arr_str1, arr_str2)
                                else:
                                    indx = path1.index(d)
                                    path1.remove(d)
                                    exit_cur = 1
                                    break
                return backtrack, rel_src, gc, map_req, an


def ter(dest, window_k, path):  # if destination is in window terminate there only
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
        if map[i] in window_k:
            l, b = rel_map[map[i]]
            dist.append(tuple([math.sqrt((l - r) ** 2 + (b - t) ** 2)]))
            cor.append(rel_map[map[i]])

    kc = min(dist)

    index = dist.index(kc)
    val = cor[index]
    c, v = val
    m = c - r

    n = v - t
    return m, n



rel_map = {}
rel_map1 = {}


def createwindow(m, n, sq):
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
                            arr.append(tuple([middle - kx, middle + jz]))
                            arr2.append(map[(o - kx, u + jz)])




    elif m <= 0 and n <= 0:
        # we can go up normally and down k+m+1
        # we can go left normally and right k+n+1 times

        for i in range(0,
                       k + m + 1):
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
        # we can go up normally and down k+m+1
        # we can go right normally and left k-n+1 times

        for i in range(0,
                       k + m + 1):
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
    rel_map = dict(rel_map)

    rel_map1 = zip(arr, arr2)
    rel_map1 = dict(rel_map1)

    return window_k, rel_map, rel_map1


p = 0
cal_m = 0
first = 1
l = 0
ret = 0
first = 0
lp = 0
while ret != 1:
    lp = lp + 1
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
    src = map1[backtrack[0]]
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
        f = 0
        fg = 0
        while exit != 1:
            if fg == 0:
                stop = rel(rel_src, rel_src1)
                fg = 1
            else:
                stop = rel(rel_src, rel_map[backtrack[0]])

            tr = tr + 1
            indx = -1
            if stop == 1:
                exit = 1
                reversed_backtrack = reverse(backtrack)
                zx = backtrack[0]
                backtrack1 = backtrack[1:]
                for i in backtrack1:
                    path.append(i)
                    p_b.append(i)
                path.append(zx)
                path.append('/')
                p_b.append(zx)
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

                        for d in reverse(path1[0:-(len(path1) - ind)]):
                            if d != '/':
                                if d != el_y and (d, el_y) not in an:
                                    arr_str1.append(tuple([d, el_y]))
                                    arr_str2.append(abs(target_G[d][el_y]))
                                    map_req = zip(arr_str1, arr_str2)
                                else:
                                    indx = path1.index(d)
                                    path1.remove(d)
                                    break

            else:
                if f == 0:
                    rel_src = rel_src1
                    f = 1
                backtrack, rel_src, t, arr_str1, arr_str2, an = maxim(rel_src, 1, 0, t, backtrack, rel_src1,
                                                                      an)
                map_req = zip(arr_str1, arr_str2)
