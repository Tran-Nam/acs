n_row, n_col = input().split(' ')
n_row, n_col = int(n_row), int(n_col)
# import numpy as np

map = []
for _ in range(n_row):
    map.append(input())

def find(s):
    loc = []
    for i in range(n_row):
        for j in range(n_col):
            if map[i][j] == s:
                loc.append((i, j))
    return loc

def calc_dist(s, p_seat):
    max_p = 100
    for p in p_seat:
        dist = abs(s[0] - p[0]) + abs(s[1] - p[1])
        if dist < max_p:
            max_p = dist 
    return max_p

def step(p, seat):
    # print('SEAT', seat)
    i, j = p
    pos_step = [
        (i-1, j),
        (i+1, j),
        (i, j-1),
        (i, j+1)
    ]
    # print(pos_step)
    remove_idx = []
    for k, (i, j) in enumerate(pos_step):
        for a, b in seat:
            if (a==i and b==j) or i<0 or j<0 or i>=n_row or j>=n_col:
                remove_idx.append(k)
                break
    # print(pos_step, remove_idx)
    pos_step = [p for i, p in enumerate(pos_step) if i not in remove_idx]
    pos_step = sorted(pos_step, key=lambda x: calc_dist(x, patient_seat), reverse=True)
    return pos_step
start_loc = find('S')[0]
end_loc = find('E')[0]
empty_seat = find('#')
patient_seat = find('*')

max_p = 100

dists = []
# visited = np.zeros((n_row, n_col))
visited = []
max_C = {}
solutions = []
# end_loc = np.array(end_loc).astype('int32')
def find(s, solution):
    global dists
    solution.append(s)
    if s[0]==end_loc[0] and s[1]==end_loc[1]:
        print('a')
        # dists.append(C)
        solutions.append(solution)
        # print(dists)
        print(solutions)
    # visited.append(s)
    # print(visited)
    p_step = step(s, empty_seat+patient_seat)
    for p_loc in p_step:
        # dist = calc_dist(p_loc, patient_seat)
        # print(p_loc, dist)
        solution_p = solution.copy()
        if p_loc not in solution: # new node
            # print('NEW', p_loc)
            # C = min(C, solution)
            # find(p_loc, solution.copy())
            find(p_loc, solution_p)
        else:
            print('VISITED', p_loc)
            continue
            # C = min(max(C, max_C[p_loc]), dist)

# if len(patient_seat)==0:
#     print('safe')
# else:
res = find(start_loc, [])
if len(solutions)==0:
    print(-1)
elif len(patient_seat)==0:
    print('safe')
else:
    dists = [min([calc_dist(p, patient_seat) for p in solution]) for solution in solutions]
    print(max(dists))