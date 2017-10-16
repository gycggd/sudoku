# coding=utf-8
import numpy as np
import cv2
import os
import shutil
from PIL import Image


## Solve Every Sudoku Puzzle

## See http://norvig.com/sudoku.html

## Throughout this program we have:
##   r is a row,    e.g. 'A'
##   c is a column, e.g. '3'
##   s is a square, e.g. 'A3'
##   d is a digit,  e.g. '9'
##   u is a unit,   e.g. ['A1','B1','C1','D1','E1','F1','G1','H1','I1']
##   grid is a grid,e.g. 81 non-blank chars, e.g. starting with '.18...7...
##   values is a dict of possible values, e.g. {'A1':'12349', 'A2':'8', ...}

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a + b for a in A for b in B]


digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
squares = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])
units = dict((s, [u for u in unitlist if s in u])
             for s in squares)
peers = dict((s, set(sum(units[s], [])) - {s})
             for s in squares)


################ Unit Tests ################

def test():
    "A set of tests that must pass."
    assert len(squares) == 81
    assert len(unitlist) == 27
    assert all(len(units[s]) == 3 for s in squares)
    assert all(len(peers[s]) == 20 for s in squares)
    assert units['C2'] == [['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2'],
                           ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
                           ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']]
    assert peers['C2'] == {'A2', 'B2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2', 'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
                           'C9', 'A1', 'A3', 'B1', 'B3'}
    print 'All tests pass.'


################ Parse a Grid ################

def parse_grid(grid):
    """Convert grid to a dict of possible values, {square: digits}, or
    return False if a contradiction is detected."""
    ## To start, every square can be any digit; then assign values from the grid.
    values = dict((s, digits) for s in squares)
    for s, d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False  ## (Fail if we can't assign d to square s.)
    return values


def grid_values(grid):
    "Convert grid into a dict of {square: char} with '0' or '.' for empties."
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))


################ Constraint Propagation ################

def assign(values, s, d):
    """Eliminate all the other values (except d) from values[s] and propagate.
    Return values, except return False if a contradiction is detected."""
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False


def eliminate(values, s, d):
    """Eliminate d from values[s]; propagate when values or places <= 2.
    Return values, except return False if a contradiction is detected."""
    if d not in values[s]:
        return values  ## Already eliminated
    values[s] = values[s].replace(d, '')
    ## (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
    if len(values[s]) == 0:
        return False  ## Contradiction: removed last value
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    ## (2) If a unit u is reduced to only one place for a value d, then put it there.
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False  ## Contradiction: no place for this value
        elif len(dplaces) == 1:
            # d can only be in one place in unit; assign it there
            if not assign(values, dplaces[0], d):
                return False
    return values


################ Display as 2-D grid ################

def display(values):
    "Display these values as a 2-D grid."
    width = 1 + max(len(values[s]) for s in squares)
    line = '+'.join(['-' * (width * 3)] * 3)
    for r in rows:
        print ''.join(values[r + c].center(width) + ('|' if c in '36' else '')
                      for c in cols)
        if r in 'CF': print line
    print


################ Search ################

def solve(grid):
    return search(parse_grid(grid))


def search(values):
    "Using depth-first search and propagation, try all possible values."
    if values is False:
        return False  ## Failed earlier
    if all(len(values[s]) == 1 for s in squares):
        return values  ## Solved!
    ## Chose the unfilled square s with the fewest possibilities
    n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    return some(search(assign(values.copy(), s, d))
                for d in values[s])


################ Utilities ################

def some(seq):
    "Return some element of seq that is true."
    for e in seq:
        if e: return e
    return False


## References used:
## http://www.scanraid.com/BasicStrategies.htm
## http://www.sudokudragon.com/sudokustrategy.htm
## http://www.krazydad.com/blog/2005/09/29/an-index-of-sudoku-strategies/
## http://www2.warwick.ac.uk/fac/sci/moac/currentstudents/peter_cock/python/sudoku/

path_all_numbers = './numbers'
path_tmp = './tmp'
path_contour = os.path.join(path_tmp, 'contour')
path_number = os.path.join(path_tmp, 'number')
path_resized_number = os.path.join(path_tmp, 'resized_number')

if os.path.exists(path_tmp):
    shutil.rmtree(path_tmp)
os.mkdir(path_tmp)
os.mkdir(path_contour)
os.mkdir(path_number)
os.mkdir(path_resized_number)

for f in os.listdir(path_all_numbers):
    img_pil = Image.open(os.path.join(path_all_numbers, f))
    img_origin = np.array(img_pil)
    img = img_origin.copy()
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)

    image, contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

    height, width = img.shape[:2]

    list = []
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)

        if h > (height / 4):
            list.append([x, y, w, h])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(path_contour, 'contour_' + f), img)

    list_sorted = sorted(list, cmp=lambda p1, p2: p1[1] - p2[1] if abs(p1[1] - p2[1]) > height / 4 else p1[0] - p2[0])
    list_sorted = list_sorted[-1:] + list_sorted[:-1]

    for idx, (x, y, w, h) in enumerate(list_sorted):
        cv2.imwrite(os.path.join(path_number, str(idx) + '_' + f.split('.')[0] + '.jpg'),
                    img_origin[y:y + h, x:x + w])
        # cv2.imshow(f.split('.')[0] + '_' + str(idx), img[y:y + h, x:x + w])

images = []
labels = []

for f in os.listdir(path_number):
    img_pil = Image.open(os.path.join(path_number, f))
    img = np.array(img_pil)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (20, 40))
    img = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2)
    cv2.imwrite(os.path.join(path_resized_number, f), img)
    normalized_img = img / 255
    images.append(normalized_img.flatten())
    labels.append(int(f[0]))

images = np.array(images, np.float32)
labels = np.array(labels, np.float32)

model = cv2.ml.KNearest_create()
model.train(images, cv2.ml.ROW_SAMPLE, labels)

board_origin = Image.open('./sudoku.png')
board_origin = np.array(board_origin)

if board_origin.ndim == 3:
    board_origin = cv2.cvtColor(board_origin, cv2.COLOR_BGR2RGB)
    board_gray = cv2.cvtColor(board_origin, cv2.COLOR_RGB2GRAY)
else:
    board_gray = board_origin

board_gray = cv2.GaussianBlur(board_gray, (5, 5), 0)
board_thresh = cv2.adaptiveThreshold(board_gray, 255, 1, 1, 5, 2)
cv2.imshow('board_thresh', board_thresh)

_, contours, hierarchy = cv2.findContours(board_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_contour = contours[0]
max_rect = cv2.boundingRect(contours[0])
max_idx = 0

for idx, contour in enumerate(contours):
    [x, y, w, h] = cv2.boundingRect(contour)
    if w * h > max_rect[2] * max_rect[3]:
        max_contour = contour
        max_rect = [x, y, w, h]
        max_idx = idx

grid_points = []
unit_w = max_rect[2] / 18
unit_h = max_rect[3] / 18
for i in range(9):
    for j in range(9):
        grid_points.append((max_rect[0] + (2 * i + 1) * unit_w, max_rect[1] + 2 * unit_h * j))
for i in range(9):
    for j in range(9):
        grid_points.append((max_rect[0] + 2 * unit_w * i, max_rect[1] + (2 * j + 1) * unit_h))

# for p in grid_points:
#     cv2.circle(board, p, 3, (0, 255, 0), 3)

cv2.drawContours(board_origin, [max_contour], 0, (0, 255, 0), 3)

boxes = []

candidates = []
for i in range(len(hierarchy[0])):
    if hierarchy[0][i][3] == max_idx:
        candidates.append(i)

while len(candidates) > 0:
    c = candidates[0]
    candidates.pop(0)
    rect = cv2.boundingRect(contours[c])

    if rect[2] > max_rect[2] / 10 or rect[3] > max_rect[3] / 10:
        for j in range(len(hierarchy[0])):
            if hierarchy[0][j][3] == c:
                candidates.append(j)
    elif (rect[3] > max_rect[2] / 20 or rect[3] > max_rect[3] / 18) and \
            (rect[3] < max_rect[2] / 9 or rect[3] < max_rect[3] / 9):
        cv2.drawContours(board_origin, contours, c, (0, 0, 255), 3)
        boxes.append(contours[c])

print '# of grid already filled: ' + str(len(boxes))
cv2.imshow("img", board_origin)
cv2.waitKey(5000)

height, width = board_gray.shape[:2]
box_w = max_rect[2] / 9
box_h = max_rect[3] / 9

# print boxes
sudoku = np.zeros((9, 9), np.int32)
prefilled = []

for box in boxes:
    x, y, w, h = cv2.boundingRect(box)
    cv2.rectangle(board_origin, (x - 1, y - 1), (x + w + 1, y + h + 1), (0, 0, 255), 2)
    cv2.drawContours(board_origin, [box], 0, (0, 255, 0), 1)

    number_img = board_gray[y:y + h, x:x + w]
    number_resized = cv2.resize(number_img, (20, 40))
    number_thresh = cv2.adaptiveThreshold(number_resized, 255, 1, 1, 11, 2)
    normalized_number = number_thresh / 255.

    sample1 = normalized_number.reshape((1, 800))
    sample1 = np.array(sample1, np.float32)

    _, results, neigh_resp, dists = model.findNearest(sample1, 1)
    number = int(results.ravel()[0])

    cv2.putText(board_origin, str(number), (x, y), 3, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    sudoku[int((y - max_rect[1]) / box_h)][int((x - max_rect[0]) / box_w)] = number
    prefilled.append((int((y - max_rect[1]) / box_h), int((x - max_rect[0]) / box_w)))

sudoku_str = ''

for i in range(9):
    for j in range(9):
        if sudoku[i][j] != 0:
            sudoku_str += str(sudoku[i][j])
        else:
            sudoku_str += '.'

solution_dic = solve(sudoku_str)

if solution_dic:
    for key in solution_dic:
        sudoku[ord(key[0]) - ord('A')][int(key[1]) - 1] = solution_dic[key]

    for i in range(9):
        for j in range(9):
            if (j, i) in prefilled:
                continue
            x = int((i + 0.25) * box_w)
            y = int((j + 0.75) * box_h)
            cv2.putText(board_origin, str(sudoku[j][i]), (max_rect[0] + x, max_rect[1] + y), 3, 2, (0, 0, 0), 2,
                        cv2.LINE_AA)
else:
    print "Unable to solve this Sudoku"

print(sudoku)

cv2.imwrite('./solution.png', board_origin)
cv2.imshow("img", board_origin)
cv2.waitKey(0)
