# ADVENT OF CODE DAY 12 SOLUTION (BFS/GRAPH TRAVERSAL)
import math
from functools import cmp_to_key
from collections import deque
# grid positions listed as y, x instead of x, y. This is the case until the very end.
queue = deque()
visited = {}
# will be a list of lists, with the inidces as coordinates, i.e. field[4][3] is y=4, x=3 (zero-indexed)
field = []
START_POINT = (0, 0)  # set these values later
END_POINT = (0, 0)

def process():
    numOfLines = 41  # get this from an opening input() statement
    for count in range(numOfLines):
        line = input()
        # where operation works on the character AND its position
        field.append([operation(letter, count, inner)
                     for inner, letter in enumerate(line)])
    BFS()


def BFS():
    # this searches backwards, finding a path from the end to the start.
    queue.append(END_POINT)
    # Change this to START_POINT to search forwards
    while (len(queue)) > 0:
        for spot in searchAround(queue[0]):
            visited[spot] = queue[0]
            queue.append(spot)
            if spot == START_POINT:
                traverse(spot)
        queue.popleft()


def traverse(start):  # start is a y,x tuple. Since we searched backwards,
    # we traverse forwards to build the final path
    currentLocation = start
    path = [start]
    while (currentLocation != END_POINT):
        currentLocation = visited[currentLocation]
        path.append(currentLocation)
    # the path, in x/y coordinates
    print([(spot[1], spot[0]) for spot in path])
    # subract one for number of steps taken
    print("Shortest path length:" + str(len(path)) + " nodes visited.")
    exit()


def searchAround(centre):  # centre is an x,y coordinate pair as a tuple
    GRID_HEIGHT = len(field)
    GRID_WIDTH = len(field[0])
    output = []
    if centre[0] > 0:  # search to the left, if not on the left edge
        # if condition() depends on the current centre, must also pass its position in
        out = check_and_append(centre[0] - 1, centre[1], centre[0], centre[1])
        if out is not None:
            output.append(out)
        # if centre[1] > 0: #if diagonals are allowed, these need to be a thing for all 4 diagonals, only one shown here (this checks down-right)
        # check_and_append(centre[0]- 1, centre[1] - 1)
    if centre[0] < GRID_HEIGHT - 1:  # search to the right, if not on the right edge
        out = check_and_append(centre[0] + 1, centre[1], centre[0], centre[1])
        if out is not None:
            output.append(out)
    if centre[1] > 0:
        out = check_and_append(centre[0], centre[1] - 1, centre[0], centre[1])
        if out is not None:
            output.append(out)
    if centre[1] < GRID_WIDTH - 1:
        out = check_and_append(centre[0], centre[1] + 1, centre[0], centre[1])
        if out is not None:
            output.append(out)
    return output


def check_and_append(x, y, centrex, centrey):
    if (not visited.get((x, y))):  # if we haven't visited (x,y) yet, consider this path/case
        # where condition() may depend on currently visitING x, y, or the original centre from which we came.
        if condition(x, y, centrex, centrey):
            # condition should be true iff the spot we're checking could be part of the path.
            return (x, y)


def condition(y, x, centrey, centrex):  # specific to the advent of code solution,
    # will need to change this to match our problem.
    return field[y][x] >= field[centrey][centrex] - 1


process()
# END ADVENT OF CODE SOLUTION

# DJIKSTRA'S ALGORITHM


def minimum(dicti):
    min_key = list(dicti.keys())[0]
    for i in list(dicti.keys)[1:]:
        if dicti[i] < dicti[min_key]:
            min_key = i
    return min_key


def dijkstra(airports, lines, start, end):
    unexplored = {airport: float('inf') for airport in airports}
    unexplored[start] = 0
    while len(unexplored) != 0:
        explore = minimum(unexplored)
        if explore == end:
            break
        else:
            for path in lines.items():
                if path[0][0] == explore:
                    if path[0][1] in unexplored.keys():
                        check_time = unexplored[path[0][0]] + path[1]
                        if check_time < unexplored[path[0][1]]:
                            unexplored[path[0][1]] = check_time
                elif path[0][1] == explore:
                    if path[0][0] in unexplored.keys():
                        check_time = unexplored[path[0][1]] + path[1]
                        if check_time < unexplored[path[0][0]]:
                            unexplored[path[0][0]] = check_time
            del unexplored[explore]
    return (unexplored[explore])


airports = ['A', 'B', 'C', 'D', 'E']
lines = {
    ('A', 'B'): 4,
    ('A', 'C'): 2,
    ('B', 'C'): 1,
    ('B', 'D'): 2,
    ('C', 'D'): 4,
    ('C', 'E'): 5,
    ('E', 'D'): 1,
}
start = 'A'
end = 'D'

print(dijkstra(airports, lines, start, end))


# where the first line goes from (x1, y1) to (x2, y2)
def two_lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # and the second line goes from (x3, y3) to (x4, y4). works even if one line is vertical
    try:
        px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / \
            ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))  # intersection X coordinate
        py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / \
            ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))  # intersection Y coordinate
        # answer[0] is "do they intersect within bounds",
        return (x1 < px < x2) and (x3 < px < x4), (px, py)
        # answer[1] is "where do they intersect", even if it's out of bounds.
    # lines have the same slope (no single intersection)
    except ZeroDivisionError:
        return False, False  # though they may be the same line


def is_in_sorted(ls, n):  # find if n is in the sorted list ls
    return bisect_left(ls, n) < len(ls) and ls[bisect_left(ls, n)] == n
# number n is prime if it is <= 2 ** 32, odd, and passes sprp_test for a = 2, a = 7, and a = 61
# optimal setup: trial division by all primes up to 300 or so, then run this test


def sprp_test(n, a):
    d = n - 1
    s = 0
    while (d % 2 == 1):
        s += 1
        d //= 2
    test = pow(a, d, n)
    if test == 1 or test == n - 1:
        return True
    for r in range(1, s):
        test = pow(test, 2, n)
        if test == n - 1:
            return True
    return False


# where the line is (x1, y1) -> (x2, y2) and the point is at (px, py)
def left_predicate(x1, y1, x2, y2, px, py):
    return (x2 - x1) * (py - y1) - (y2 - y1)*(px - x1) > 0


##### GRAHAM's SCAN ALGORITHM#####

# A class used to store the x and y coordinates of points

class Point:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y


# A global point needed for sorting points with reference
# to the first point
p0 = Point(0, 0)

# A utility function to find next to top in a stack


def nextToTop(S):
    return S[-2]

# A utility function to return square of distance
# between p1 and p2


def distSq(p1, p2):
    return ((p1.x - p2.x) * (p1.x - p2.x) +
            (p1.y - p2.y) * (p1.y - p2.y))

# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise


def orientation(p, q, r):
    val = ((q.y - p.y) * (r.x - q.x) -
           (q.x - p.x) * (r.y - q.y))
    if val == 0:
        return 0  # collinear
    elif val > 0:
        return 1  # clock wise
    else:
        return 2  # counterclock wise

# A function used by cmp_to_key function to sort an array of
# points with respect to the first point


def compare(p1, p2):

    # Find orientation
    o = orientation(p0, p1, p2)
    if o == 0:
        if distSq(p0, p2) >= distSq(p0, p1):
            return -1
        else:
            return 1
    else:
        if o == 2:
            return -1
        else:
            return 1

# Prints convex hull of a set of n points.


def convexHull(points, n):

    # Find the bottommost point
    ymin = points[0].y
    min = 0
    for i in range(1, n):
        y = points[i].y

        # Pick the bottom-most or choose the left
        # most point in case of tie
        if ((y < ymin) or
                (ymin == y and points[i].x < points[min].x)):
            ymin = points[i].y
            min = i

    # Place the bottom-most point at first position
    points[0], points[min] = points[min], points[0]

    # Sort n-1 points with respect to the first point.
    # A point p1 comes before p2 in sorted output if p2
    # has larger polar angle (in counterclockwise
    # direction) than p1
    p0 = points[0]
    points = sorted(points, key=cmp_to_key(compare))

    # If two or more points make same angle with p0,
    # Remove all but the one that is farthest from p0
    # Remember that, in above sorting, our criteria was
    # to keep the farthest point at the end when more than
    # one points have same angle.
    m = 1  # Initialize size of modified array
    for i in range(1, n):

        # Keep removing i while angle of i and i+1 is same
        # with respect to p0
        while ((i < n - 1) and
               (orientation(p0, points[i], points[i + 1]) == 0)):
            i += 1

        points[m] = points[i]
        m += 1  # Update size of modified array

    # If modified array of points has less than 3 points,
    # convex hull is not possible
    if m < 3:
        return

    # Create an empty stack and push first three points
    # to it.
    S = []
    S.append(points[0])
    S.append(points[1])
    S.append(points[2])

    # Process remaining n-3 points
    for i in range(3, m):

        # Keep removing top while the angle formed by
        # points next-to-top, top, and points[i] makes
        # a non-left turn
        while ((len(S) > 1) and
               (orientation(nextToTop(S), S[-1], points[i]) != 2)):
            S.pop()
        S.append(points[i])

    # Now stack has the output points,
    # print contents of stack
    while S:
        p = S[-1]
        print("(" + str(p.x) + ", " + str(p.y) + ")")
        S.pop()


# Driver Code
input_points = [(0, 3), (1, 1), (2, 2), (4, 4),
                (0, 0), (1, 2), (3, 1), (3, 3)]
points = []
for point in input_points:
    points.append(Point(point[0], point[1]))
n = len(points)
convexHull(points, n)
# END GRAHAM SCAN ALGORITHM

##UNION FIND
class TreeNode:
    def __init__(self):
        self.parent = self
        self.size = 1

    def find_ancestor(self):
    #performs two tasks: finding the ancestor and flattening the tree
    #the tree flattening makes it faster to find ancestors in future
        root = self
        x = self
        while root.parent != root:
            root = root.parent
            #root is now the "representative" of the whole set, at the top
        while x.parent != root:
            new_parent = x.parent
            x.parent = root
            x = new_parent
        return root

    def merge(self, u2):
        u1_rep = self.find_ancestor()
        u2_rep = u2.find_ancestor()
        if u1_rep == u2_rep:
            return #no need to merge, they're in the same network already
        largest_rep = u1_rep if u1_rep.size > u2_rep.size else u2_rep
        smallest_rep = u2_rep if largest_rep == u1_rep else u1_rep
        smallest_rep.parent = largest_rep #merge the smaller network into the larger one
        largest_rep.size += smallest_rep.size
        del smallest_rep.size #maybe a memory optimization, smaller rep's size never used

#to create a relationship between u1 and u2, use u1.merge(u2)
#to see if u1 and u2 are in the same network, compare u1.find_ancestor() == u2.find_ancestor()

#CEO Problem Solution from last year (mostly): Uses Trees and Dynamic Programming
#passes most test cases so it's an OK example
class TreeNode:
    def __init__(self, val):
        self.childlist = []
        self.val = val
        self.parent = None
    def addchild(self, child):
        self.childlist.append(child)
    def setparent(self, parent):
        self.parent = parent
        self.parent.addchild(self)
    def getgrandchildren(self):
        output = []
        for kid in self.childlist:
            output.extend(kid.childlist)
        #like append but can append all list elements to the end of the list
        return output

numLines = int(input())
emps = []
emps_tuples = []
for count in range(numLines):  # first pass
    line = [int(i) for i in input().split(sep=" ")]
    emps_tuples.append((line[0], line[1]))
    emps.append(TreeNode(line[0], count))

for count in range(1, numLines):  # second pass, everything but the root
    line = emps_tuples[count]
    emps[count].setparent(emps[line[1]])
del emps_tuples

sols = {}
def maximize(root):
    if root.childlist == []:
        return root.val
    if root.empnum in sols:
        return sols[root.empnum]
    included = root.val + sum([maximize(i) for i in root.getgrandchildren()])
    excluded = sum(maximize(i) for i in root.childlist)
    sols[root.empnum] = max(included, excluded)
    return sols[root.empnum]
print(maximize(emps[0]))

#powerset of an iterable
from itertools import chain, combinations
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

#EDIT DISTANCE
Suppose that we are given a string x of length n and a string y of length m,
and we want to calculate the edit distance between x and y. To solve the problem,
we define a function distance(a, b) that gives the edit distance between prefixes
x[0 . . . a] and y[0 . . . b]. Thus, using this function, the edit distance between x and
y equals distance(n − 1, m − 1).
We can calculate values of distance as follows:
distance(a, b) = min(distance(a, b − 1) + 1,
distance(a − 1, b) + 1,
distance(a − 1, b − 1) + cost(a, b)).
Here cost(a, b) = 0 if x[a] = y[b], and otherwise cost(a, b) = 1.
# for a question like the CEO question where we;re given the parent,
# put them in a list, and have a list of tree nodes, it's not ideal but what else do you do?


# Prime factorization of any integer (very slow for large numbers)


def primeFactors(n):
    while n % 2 == 0:
        print(2)
        n = n // 2
    for i in range(3, int(math.sqrt(n))+1, 2):
        while n % i == 0:
            print(i),
            n = n // i
    if n > 2:
        print(n)


def strange_input_example():
    # for when the problem diesn't say how many cases/lines there are
    for line in sys.stdin:
        try:
            process(line)

# finds subsets of an array that add to a number N


def get_subsets_adding_to_n(array, n):
    dp = [1] + [0]*n
    curr = 0
    for i in range(0, len(array)):
        curr += array[i]
        for j in range(min(n, curr), array[i]-1, -1):
            dp[j] += dp[j - array[i]]
    return dp[-1]


# gets the optimal number of cuts to turn a rectangle into squares
# note: this won't be fast enough for large numbers (over 100 or so)
# unless they share some factors, but it's the best we got
memo = {}


def optimal_rectangle_cut(i, j):
    import math
    # base cases: o-width line or already a square
    if (i == j) or (i <= 0) or (j <= 0):
        return 0
    gcd = math.gcd(i, j)
    width = max(i, j) // gcd
    height = min(i, j) // gcd
    if (height == 1):  # remember that we just took the gcd
        return width - 1
    if (height, width) in memo:
        return memo[(height, width)]
    hcut = 1 + min([optimal_rectangle_cut(width, count) + optimal_rectangle_cut(
        width, height - count) for count in range(1, height // 2 + 1)])
    vcut = 1 + min([optimal_rectangle_cut(count, height) + optimal_rectangle_cut(
        width - count, height) for count in range(1, width // 2 + 1)])
    memo[(height, width)] = min(hcut, vcut)
    return memo[(height, width)]

# where W is the max weight of the backpack, wt is an array of weights, and val is an array of values
def knapSack(W, wt, val):
    n = len(val)
    table = [[0 for x in range(W + 1)] for x in range(n + 1)]

    for i in range(n + 1):
        for j in range(W + 1):
            if i == 0 or j == 0:
                table[i][j] = 0
            elif wt[i-1] <= j:
                table[i][j] = max(val[i-1]
                                  + table[i-1][j-wt[i-1]],  table[i-1][j])
            else:
                table[i][j] = table[i-1][j]
    return table[n][W]

# Function to Check if a substring is a palindrome


def is_palindrome(string, i, j):
    while i < j:
        if string[i] != string[j]:
            return False
        i += 1
        j -= 1
    return True

# Function to find the minimum number of cuts needed for palindrome partitioning


def min_pal_partition(string, i, j):
    # Base case: If the substring is empty or a palindrome, no cuts needed
    if i >= j or is_palindrome(string, i, j):
        return 0
    ans = 10**70  # absurdly high number
    # Iterate through all possible partitions and find the minimum cuts needed
    for k in range(i, j):
        count = min_pal_partition(string, i, k) + \
            min_pal_partition(string, k + 1, j) + 1
        ans = min(ans, count)
    return ans

# Line and Point functions from cp4

INF = 10**9
EPS = 1e-6
def DEG_to_RAD(d):
    return d*math.pi/180.0
def RAD_to_DEG(r):
    return r*180.0/math.pi
class point_i:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
class point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)
    def __eq__(self, other):
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)
def dist(p1, p2):
    return math.hypot(p1.x-p2.x, p1.y-p2.y)
# Rotate a point by theta
def rotate(p, theta):
    rad = DEG_to_RAD(theta)
    x = p.x * math.cos(rad) - p.y * math.sin(rad)
    y = p.x * math.sin(rad) + p.y * math.cos(rad)
    return point(x, y)
class line:
    def __init__(self):
        self.a = 0
        self.b = 0
        self.c = 0
def pointsToLine(p1, p2, l):
    if abs(p1.x - p2.x) < EPS:
        l.a, l.b, l.c = 1.0, 0.0, -p1.x
    else:
        a = -(p1.y - p2.y) / (p1.x - p2.x)
        l.a, l.b, l.c = a, 1.0, -(a * p1.x) - p1.y
class line2:
    def __init__(self):
        self.m = 0
        self.c = 0
def pointsToLine2(p1, p2, l):
    if p1.x == p2.x:
        l.m = INF
        l.c = p1.x
        return 0
    else:
        l.m = (p1.y - p2.y) / (p1.x - p2.x)
        l.c = p1.y - l.m * p1.x
        return 1

# Check if two lines are parallel
def areParallel(l1, l2):
    return math.isclose(l1.a, l2.a) and math.isclose(l1.b, l2.b)
# Check if two lines are same
def areSame(l1, l2):
    return areParallel(l1, l2) and math.isclose(l1.c, l2.c)
def areIntersect(l1, l2, p):
    if areParallel(l1, l2):
        return False
    p.x = (l2.b * l1.c - l1.b * l2.c) / (l2.a * l1.b - l1.a * l2.b)
    if not math.isclose(l1.b, 0.0):
        p.y = -(l1.a * p.x + l1.c)
    else:
        p.y = -(l2.a * p.x + l2.c)
    return True

class vec:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
def toVec(a, b):
    return vec(b.x-a.x, b.y-a.y)
def scale(v, s):
    return vec(v.x*s, v.y*s)
def translate(p, v):
    return point(p.x+v.x, p.y+v.y)
def pointSlopeToLine(p, m, l):
    l.a, l.b = -m, 1
    l.c = -((l.a * p.x) + (l.b * p.y))

def closestPoint(l, p, ans):
    if math.isclose(l.b, 0.0):
        ans.x, ans.y = -l.c, p.y
        return
    if math.isclose(l.a, 0.0):
        ans.x, ans.y = p.x, -l.c
        return
    perpendicular = line()
    pointSlopeToLine(p, 1.0/l.a, perpendicular)
    areIntersect(l, perpendicular, ans)


def reflectionPoint(l, p, ans):
    b = point()
    closestPoint(l, p, b)
    v = toVec(p, b)
    ans.x, ans.y = p.x + 2 * v.x, p.y + 2 * v.y

# Dot Product
def dot(a, b):
    return a.x * b.x + a.y * b.y
def norm_sq(v):
    return v.x * v.x + v.y * v.y
def angle(a, o, b):
    oa = toVec(o, a)
    ob = toVec(o, b)
    return math.acos(dot(oa, ob) / math.sqrt(norm_sq(oa) * norm_sq(ob)))

def distToLine(p, a, b, c):
    ap = toVec(a, p)
    ab = toVec(a, b)
    u = dot(ap, ab) / norm_sq(ab)
    s = scale(ab, u)
    c.x, c.y = a.x+s.x, a.y+s.y
    return dist(p, c)
def distToLineSegment(p, a, b, c):
    ap = toVec(a, p)
    ab = toVec(a, b)
    u = dot(ap, ab) / norm_sq(ab)
    if u < 0.0:
        c.x, c.y = a.x, a.y
        return dist(p, a)
    if u > 1.0:
        c.x, c.y = b.x, b.y
        return dist(p, b)
    return distToLine(p, a, b, c)
def cross(a, b):
    return a.x * b.y - a.y * b.x
def ccw(p, q, r):
    return cross(toVec(p, q), toVec(p, r)) > -EPS
def collinear(p, q, r):
    return abs(cross(toVec(p, q), toVec(p, r))) < EPS

def gcdExtended(a, b):
    # Base Case
    if a == 0 :
        return b,0,1
    gcd,x1,y1 = gcdExtended(b%a, a)
    # Update x and y using results of recursive
    # call
    x = y1 - (b//a) * x1
    y = x1
    return gcd,x,y
	#returns ax+by such that ax+by=gcd

#boggle algorithm, useful for 2-D grid traversal
#note that the boggle grid is 4x4
def find_words(self, board):
    def dfs(row, col, path, node, word):
        # Helper function for depth-first search to find words on the board
        if not (0 <= row < 4) or not (0 <= col < 4) or (row, col) in path:
            return
        letter = board[row][col]
        if letter in node:
            path.append((row, col))
            node = node[letter]
            word += letter
            if '*' in node and len(word) > 2:
                self.validated.add(word)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    dfs(row + dr, col + dc, path, node, word)
            path.pop()
    # Iterate through the entire board to find words
    for r in range(4):
        for c in range(4):
            dfs(r, c, [], self.root, '')
