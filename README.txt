Water Jug
from collections import deque
def water_jug_BFS(x, y, z):
    visited = set()
    queue = deque([(0, 0)])    
    while queue:
        jug_a, jug_b = queue.popleft()
       if jug_a == z or jug_b == z or jug_a + jug_b == z:
            return True
      if (jug_a, jug_b) in visited:
            continue
    visited.add((jug_a, jug_b))
       # Fill jug A
        if jug_a < x:
    queue.append((x, jug_b))
       # Fill jug B
        if jug_b < y:
         queue.append((jug_a, y))
        
        # Empty jug A
        if jug_a > 0:
 queue.append((0, jug_b))
        # Empty jug B
        if jug_b > 0:
            queue.append((jug_a, 0))
        # Pour from A to B
        if jug_a + jug_b >= y:
            queue.append((jug_a - (y - jug_b), y))
        else:
            queue.append((0, jug_a + jug_b))
        # Pour from B to A
        if jug_a + jug_b >= x:
            queue.append((x, jug_b - (x - jug_a)))
        else:
            queue.append((jug_a + jug_b, 0))
    return False
x = 2
y = 1 
z = 4 
if water_jug_BFS(x, y, z):
print(f'You can measure {z} liters of water using {x}-liter and {y}-liter jugs.')
else:
    print(f'You cannot measure {z} liters of water using {x}-liter and {y}-liter jugs.')
=================================Travelling saleman
dst=[]
def travel(g, v, pos, n, count, cost):
    if(count==n and g[pos][s]):
        cost+=g[pos][s]
        dst.append(cost)
        return
    for i in range(0,n):
        if(v[i]==False and g[pos][i]):
            v[i]=True
travel(g,v,i,n,count+1,cost+g[pos][i])
v[i]=False
n=4
g=[[0, 10, 15, 20],[10, 0, 35, 25],[15, 35, 0, 30],[20, 25, 30, 0]]
s=int(input("Enter a number between 1 and 4: "))
v=[False for i in range(0,n)]
s-=1
v[s]=True
travel(g,v,s,n,1,0)
print(dst)
print(min(dst))
==================================
BFS 
import sys
import copy
q = []
visited = []
def compare(s,g):
    if s==g:
        return(1)
    else:
        return(0)
def find_pos(s):
    for i in range(3):
        for j in range(3):
            if s[i][j] == 0:
                return([i,j])
def up(s,pos):
    i = pos[0]
    j = pos[1]
    if i > 0:
        temp = copy.deepcopy(s)
        temp[i][j] = temp[i-1][j]
        temp[i-1][j] = 0
        return (temp)
    else:
        return (s)
def down(s,pos):
    i = pos[0]
    j = pos[1]
    if i < 2:
        temp = copy.deepcopy(s)
        temp[i][j] = temp[i+1][j]
        temp[i+1][j] = 0
        return (temp)
    else:
        return (s)
def right(s,pos):
    i = pos[0]
    j = pos[1]
    if j < 2:
        temp = copy.deepcopy(s)
        temp[i][j] = temp[i][j+1]
 temp[i][j+1] = 0
        return (temp)
    else:
        return (s)
def left(s,pos):
    i = pos[0]
    j = pos[1]
    if j > 0:
        temp = copy.deepcopy(s)
        temp[i][j] = temp[i][j-1]
        temp[i][j-1] = 0
        return (temp)
    else:
        return (s)
def enqueue(s,val):
    global q
    q = q + [(val,s)]
def heuristic(s,g):
    d = 0
    for i in range(3):
        for j in range(3):
            if s[i][j] != g[i][j]:
                d += 1
    return d
def dequeue():
    global q
    global visited
    q.sort()
    visited = visited + [q[0][1]]
    elem = q[0][1]
    del q[0]
    return (elem)
def search(s,g):
    curr_state = copy.deepcopy(s)
    if s == g:
        return
    global visited
    while(1):
        pos = find_pos(curr_state)
        new = up(curr_state,pos)
        if new != curr_state:
            if new == g:
                print ("found!! The intermediate states are:")
                print (visited + [g])
                return
            else:
                if new not in visited:
                    enqueue(new,heuristic(new,g))
        new = down(curr_state,pos)
        if new != curr_state:
            if new == g:
                print ("found!! The intermediate states are:")
                print (visited + [g])
return
            else:
                if new not in visited:
                    enqueue(new,heuristic(new,g))
        new = right(curr_state,pos)
        if new != curr_state:
            if new == g:
                print ("found!! The intermediate states are:")
                print (visited + [g])
                return
            else:
                if new not in visited:
                    enqueue(new,heuristic(new,g))

        new = left(curr_state,pos)
        if new != curr_state:
            if new == g:
                print ("found!! The intermediate states are:")
                print (visited + [g])
                return
            else:
                if new not in visited:
                    enqueue(new,heuristic(new,g))
        if len(q) > 0:
            curr_state = dequeue()
        else:
            print ("not found")
            return
def main():
    s = [[2,0,3],[1,8,4],[7,6,5]]
    g = [[1,2,3],[8,0,4],[7,6,5]]
    global q
    global visited
    q = q
    visited = visited + [s]
    search(s,g)

if __name__ == "__main__":
    main()

==================================
HILL CLIMB
import sys
import copy
curr_min = sys.maxsize
q = []
visited = []
def compare(s,g):
    if s==g:
        return(1)
    else:
        return(0)
def find_pos(s):
    for i in range(3):
        for j in range(3):
            if s[i][j] == 0:
                return([i,j])

def up(s,pos):
    i = pos[0]
    j = pos[1]
    if i > 0:
        temp = copy.deepcopy(s)
        temp[i][j] = temp[i-1][j]
        temp[i-1][j] = 0
        return (temp)
    else:
        return (s)
def down(s,pos):
    i = pos[0]
    j = pos[1]
    if i < 2:
        temp = copy.deepcopy(s)
        temp[i][j] = temp[i+1][j]
        temp[i+1][j] = 0
        return (temp)
    else:
        return (s)

def right(s,pos):
    i = pos[0]
    j = pos[1]
    if j < 2:
        temp = copy.deepcopy(s)
        temp[i][j] = temp[i][j+1]
        temp[i][j+1] = 0
        return (temp)
    else:
        return (s)
def left(s,pos):
    i = pos[0]
    j = pos[1]
    if j > 0:
        temp = copy.deepcopy(s)
        temp[i][j] = temp[i][j-1]
        temp[i][j-1] = 0
        return (temp)
    else:
        return (s)

def enqueue(s):
    global q
    q = q + [s]
def heuristic(s,g):
    d = 0
    for i in range(len(s)):
        for j in range(len(s[0])):
            if s[i][j] != g[i][j]:
                d += 1
    return d
def dequeue(g):
    h = []
    global q
    global visited
    global curr_min
    for i in range(len(q)):
        h = h + [heuristic(q[i],g)]
    if min(h) < curr_min:
        curr_min = min(h)
        index = h.index(min(h))
        visited = visited + [q[index]]
        elem = q[index]
        q = []
        return (elem)
    else:
        print ("optimal solution found !! The intermediate states are: ")
        print (visited)
        exit()
def search(s,g):
    curr_state = copy.deepcopy(s)
    if s == g:
        return
    global visited
    while(1):
        pos = find_pos(curr_state)
        new = up(curr_state,pos)
        if new != curr_state:
            if new == g:
                print ("Goal State found !! The intermediate States are :")
                print (visited + [g])
                return
            else:
                if new not in visited:
                    enqueue(new)
        new = down(curr_state,pos)
        if new != curr_state:
            if new == g:
                print ("Goal State found !! The intermediate States are :")
                print (visited + [g])
                return
            else:
                if new not in visited:
                    enqueue(new)
        new = right(curr_state,pos)
        if new != curr_state:
            if new == g:
                print ("Goal State found !! The intermediate States are :")
                print (visited + [g])
                return
            else:
                if new not in visited:
                    enqueue(new)

        new = left(curr_state,pos)

        if new != curr_state:
            if new == g:
                print ("Goal State found !! The intermediate States are :")
                print (visited + [g])
                return
            else:
                if new not in visited:
                    enqueue(new)
        if len(q) > 0:
            curr_state = dequeue(g)
        else:
            print ("not found")
            return
def main():
    s = [[2,8,3],[1,5,4],[7,6,0]]
    g = [[1,2,7],[8,0,5],[3,4,6]]
    global q
    global visited
    q = q + [s]
    visited = visited + [s]
    search(s,g)
if __name__ == "__main__":
    main()
==================================

DECISSION TREE
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
#convert to a dataframe
df = pd.DataFrame(data.data, columns = data.feature_names)
df.head()
#create the species column

df['Species'] = data.target
df.head()
#replace this with the actual names
target = np.unique(data.target)
target_names = np.unique(data.target_names)
targets = dict(zip(target, target_names))
df['Species'] = df['Species'].replace(targets)
x = df.drop(columns="Species")
y = df["Species"]
feature_names = x.columns
labels = y.unique()
#split the dataset
from sklearn.model_selection import train_test_split

X_train, test_x, y_train, test_lab = train_test_split(x,y, test_size = 0.4,random_state = 42)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy',max_depth =3, random_state = 42)
clf.fit(X_train, y_train)
from sklearn import tree

==
As a Tree Diagram
import matplotlib.pyplot as plt
plt.figure(figsize=(30,10), facecolor ='k')
a = tree.plot_tree(clf,feature_names = feature_names, class_names = labels, rounded = True, filled = True,fontsize=14)
plt.show()
#2. As a Text-Based Diagram
from sklearn.tree import export_text
tree_rules = export_text(clf,feature_names = list(feature_names))
print(tree_rules)
# Predict Class From Test Values
test_pred_decision_tree = clf.predict(test_x)
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
confusion_matrix = metrics.confusion_matrix(test_lab,  test_pred_decision_tree)
matrix_df = pd.DataFrame(confusion_matrix)
ax = plt.axes()
sns.set(font_scale=1.3)
plt.figure(figsize=(10,7))
sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")
ax.set_title('Confusion Matrix - Decision Tree')
ax.set_xlabel("Predicted label", fontsize =15)
ax.set_xticklabels(['']+labels)
ax.set_ylabel("True Label", fontsize=15)
ax.set_yticklabels(list(labels), rotation = 0)
plt.show()

=====================================
KNN 
 Setup
import numpy as np
from sklearn import datasets
from sklearn import neighbors
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
iris = datasets.load_iris()
print(iris.keys())
n_samples, n_features = iris.data.shape
print((n_samples, n_features))
print(iris.data[0])
print(iris.target.shape)
print(iris.target)
print(iris.target_names)
x_index = 0
y_index = 1
# this formatter will label the colorbar with the correct target 
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.scatter(iris.data[:, x_index], iris.data[:, y_index],
            c=iris.target, cmap=plt.cm.get_cmap('RdYlBu', 3))
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.clim(-0.5, 2.5)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index]);
X, y = iris.data, iris.target
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X, y)
result = clf.predict([[3, 5, 4, 2],])
print(iris.target_names[result])
 

# Create color maps for 3-class classification problem, as with iris
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
def plot_iris_knn():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.figure()
    pl.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training point
    pl.scatter(X[:, 0], X[:, 1], c=y,cmap=cmap_bold)
    pl.xlabel('sepal length (cm)')
    pl.ylabel('sepal width (cm)')
    pl.axis('tight')
plot_iris_knn()
=============================
NAÏVE BAYES
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total points :", ((y_test != y_pred).sum(), X_test.shape[0]))


==================================
BEST F S
from queue import PriorityQueue
v = 14
graph = [[] for i in range(v)]
 
# Function For Implementing Best First Search
# Gives output path having lowest cost
 
 
def best_first_search(actual_Src, target, n):
    visited = [False] * n
    pq = PriorityQueue()
    pq.put((0, actual_Src))
    visited[actual_Src] = True
     
    while pq.empty() == False:
        u = pq.get()[1]
        # Displaying the path having lowest cost
        print(u, end=" ")
        if u == target:
            break
 
        for v, c in graph[u]:
            if visited[v] == False:
                visited[v] = True
                pq.put((c, v))
    print()
 
# Function for adding edges to graph
 
 
def addedge(x, y, cost):
    graph[x].append((y, cost))
    graph[y].append((x, cost))
 
 
# The nodes shown in above example(by alphabets) are
# implemented using integers addedge(x,y,cost);
addedge(0, 1, 3)
addedge(0, 2, 6)
addedge(0, 3, 5)
addedge(1, 4, 9)
addedge(1, 5, 8)
addedge(2, 6, 12)
addedge(2, 7, 14)
addedge(3, 8, 7)
addedge(8, 9, 5)
addedge(8, 10, 6)
addedge(9, 11, 1)
addedge(9, 12, 10)
addedge(9, 13, 2)
 
source = 0
target = 9
best_first_search(source, target, v)
================================== 
A*

# Python program for A* Search Algorithm
import math
import heapq

# Define the Cell class


class Cell:
    def __init__(self):
      # Parent cell's row index
        self.parent_i = 0
    # Parent cell's column index
        self.parent_j = 0
 # Total cost of the cell (g + h)
        self.f = float('inf')
    # Cost from start to this cell
        self.g = float('inf')
    # Heuristic cost from this cell to destination
        self.h = 0


# Define the size of the grid
ROW = 9
COL = 10

# Check if a cell is valid (within the grid)


def is_valid(row, col):
    return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)

# Check if a cell is unblocked


def is_unblocked(grid, row, col):
    return grid[row][col] == 1

# Check if a cell is the destination


def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

# Calculate the heuristic value of a cell (Euclidean distance to destination)


def calculate_h_value(row, col, dest):
    return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5

# Trace the path from source to destination


def trace_path(cell_details, dest):
    print("The Path is ")
    path = []
    row = dest[0]
    col = dest[1]

    # Trace the path from destination to source using parent cells
    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j
        row = temp_row
        col = temp_col

    # Add the source cell to the path
    path.append((row, col))
    # Reverse the path to get the path from source to destination
    path.reverse()

    # Print the path
    for i in path:
        print("->", i, end=" ")
    print()

# Implement the A* search algorithm


def a_star_search(grid, src, dest):
    # Check if the source and destination are valid
    if not is_valid(src[0], src[1]) or not is_valid(dest[0], dest[1]):
        print("Source or destination is invalid")
        return

    # Check if the source and destination are unblocked
    if not is_unblocked(grid, src[0], src[1]) or not is_unblocked(grid, dest[0], dest[1]):
        print("Source or the destination is blocked")
        return

    # Check if we are already at the destination
    if is_destination(src[0], src[1], dest):
        print("We are already at the destination")
        return

    # Initialize the closed list (visited cells)
    closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
    # Initialize the details of each cell
    cell_details = [[Cell() for _ in range(COL)] for _ in range(ROW)]

    # Initialize the start cell details
    i = src[0]
    j = src[1]
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j

    # Initialize the open list (cells to be visited) with the start cell
    open_list = []
    heapq.heappush(open_list, (0.0, i, j))

    # Initialize the flag for whether destination is found
    found_dest = False

    # Main loop of A* search algorithm
    while len(open_list) > 0:
        # Pop the cell with the smallest f value from the open list
        p = heapq.heappop(open_list)

        # Mark the cell as visited
        i = p[1]
        j = p[2]
        closed_list[i][j] = True

        # For each direction, check the successors
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]

            # If the successor is valid, unblocked, and not visited
            if is_valid(new_i, new_j) and is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                # If the successor is the destination
                if is_destination(new_i, new_j, dest):
                    # Set the parent of the destination cell
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
                    print("The destination cell is found")
                    # Trace and print the path from source to destination
                    trace_path(cell_details, dest)
                    found_dest = True
                    return
                else:
                    # Calculate the new f, g, and h values
                    g_new = cell_details[i][j].g + 1.0
                    h_new = calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new

                    # If the cell is not in the open list or the new f value is smaller
                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        # Add the cell to the open list
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        # Update the cell details
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j

    # If the destination is not found after visiting all cells
    if not found_dest:
        print("Failed to find the destination cell")

# Driver Code


def main():
    # Define the grid (1 for unblocked, 0 for blocked)
    grid = [
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 1]
    ]

    # Define the source and destination
    src = [8, 0]
    dest = [0, 0]

    # Run the A* search algorithm
    a_star_search(grid, src, dest)


if __name__ == "__main__":
    main()
==================================
Ao *

# Cost to find the AND and OR path
def Cost(H, condition, weight = 1):
    cost = {}
    if 'AND' in condition:
        AND_nodes = condition['AND']
        Path_A = ' AND '.join(AND_nodes)
        PathA = sum(H[node]+weight for node in AND_nodes)
        cost[Path_A] = PathA
 
    if 'OR' in condition:
        OR_nodes = condition['OR']
        Path_B =' OR '.join(OR_nodes)
        PathB = min(H[node]+weight for node in OR_nodes)
        cost[Path_B] = PathB
    return cost
 
# Update the cost
def update_cost(H, Conditions, weight=1):
    Main_nodes = list(Conditions.keys())
    Main_nodes.reverse()
    least_cost= {}
    for key in Main_nodes:
        condition = Conditions[key]
        print(key,':', Conditions[key],'>>>', Cost(H, condition, weight))
        c = Cost(H, condition, weight) 
        H[key] = min(c.values())
        least_cost[key] = Cost(H, condition, weight)            
    return least_cost
 
# Print the shortest path
def shortest_path(Start,Updated_cost, H):
    Path = Start
    if Start in Updated_cost.keys():
        Min_cost = min(Updated_cost[Start].values())
        key = list(Updated_cost[Start].keys())
        values = list(Updated_cost[Start].values())
        Index = values.index(Min_cost)
         
        # FIND MINIMIMUM PATH KEY
        Next = key[Index].split()
        # ADD TO PATH FOR OR PATH
        if len(Next) == 1:
 
            Start =Next[0]
            Path += ' = ' +shortest_path(Start, Updated_cost, H)
        # ADD TO PATH FOR AND PATH
        else:
            Path +='=('+key[Index]+') '
 
            Start = Next[0]
            Path += '[' +shortest_path(Start, Updated_cost, H) + ' + '
 
            Start = Next[-1]
            Path +=  shortest_path(Start, Updated_cost, H) + ']'
 
    return Path
         
 
# Heuristic values of Nodes  
H1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1, 'T': 3}
 
 
Conditions = {
 'A': {'OR': ['D'], 'AND': ['B', 'C']},
 'B': {'OR': ['G', 'H']},
 'C': {'OR': ['J']},
 'D': {'AND': ['E', 'F']},
 'G': {'OR': ['I']}
    
}
 
# weight
weight = 1
# Updated cost
print('Updated Cost :')
Updated_cost = update_cost(H1, Conditions, weight=1)
print('*'*75)
print('Shortest Path :\n',shortest_path('A', Updated_cost,H1))
==================================
UCS 

# Python3 implementation of above approach
 
# returns the minimum cost in a vector( if
# there are multiple goal states)
def  uniform_cost_search(goal, start):
     
    # minimum cost upto
    # goal state from starting
    global graph,cost
    answer = []
 
    # create a priority queue
    queue = []
 
    # set the answer vector to max value
    for i in range(len(goal)):
        answer.append(10**8)
 
    # insert the starting index
    queue.append([0, start])
 
    # map to store visited node
    visited = {}
 
    # count
    count = 0
 
    # while the queue is not empty
    while (len(queue) > 0):
 
        # get the top element of the
        queue = sorted(queue)
        p = queue[-1]
 
        # pop the element
        del queue[-1]
 
        # get the original value
        p[0] *= -1
 
        # check if the element is part of
        # the goal list
        if (p[1] in goal):
 
            # get the position
            index = goal.index(p[1])
 
            # if a new goal is reached
            if (answer[index] == 10**8):
                count += 1
 
            # if the cost is less
            if (answer[index] > p[0]):
                answer[index] = p[0]
 
            # pop the element
            del queue[-1]
 
            queue = sorted(queue)
            if (count == len(goal)):
                return answer
 
        # check for the non visited nodes
        # which are adjacent to present node
        if (p[1] not in visited):
            for i in range(len(graph[p[1]])):
 
                # value is multiplied by -1 so that
                # least priority is at the top
                queue.append( [(p[0] + cost[(p[1], graph[p[1]][i])])* -1, graph[p[1]][i]])
 
        # mark as visited
        visited[p[1]] = 1
 
    return answer
 
# main function
if __name__ == '__main__':
     
    # create the graph
    graph,cost = [[] for i in range(8)],{}
 
    # add edge
    graph[0].append(1)
    graph[0].append(3)
    graph[3].append(1)
    graph[3].append(6)
    graph[3].append(4)
    graph[1].append(6)
    graph[4].append(2)
    graph[4].append(5)
    graph[2].append(1)
    graph[5].append(2)
    graph[5].append(6)
    graph[6].append(4)
 
    # add the cost
    cost[(0, 1)] = 2
    cost[(0, 3)] = 5
    cost[(1, 6)] = 1
    cost[(3, 1)] = 5
    cost[(3, 6)] = 6
    cost[(3, 4)] = 2
    cost[(2, 1)] = 4
    cost[(4, 2)] = 4
    cost[(4, 5)] = 3
    cost[(5, 2)] = 6
    cost[(5, 6)] = 3
    cost[(6, 4)] = 7
 
    # goal state
    goal = []
 
    # set the goal
    # there can be multiple goal states
    goal.append(6)
 
    # get the answer
    answer = uniform_cost_search(goal, 0)
 
    # print the answer
    print("Minimum cost from 0 to 6 is = ",answer[0])
==================================
Alpha beat pruning

class MiniMax:
    # print utility value of root node (assuming it is max)
    # print names of all nodes visited during search
    def __init__(self, game_tree):
        self.game_tree = game_tree  # GameTree
        self.root = game_tree.root  # GameNode
        self.currentNode = None     # GameNode
        self.successors = []        # List of GameNodes
        return

    def minimax(self, node):
        # first, find the max value
        best_val = self.max_value(node) # should be root node of tree

        # second, find the node which HAS that max value
        #  –> means we need to propagate the values back up the
        #      tree as part of our minimax algorithm
        successors = self.getSuccessors(node)
        print "MiniMax:  Utility Value of Root Node: = " + str(best_val)
        # find the node with our best move
        best_move = None
        for elem in successors:   # —> Need to propagate values up tree for this to work
            if elem.value == best_val:
                best_move = elem
                break

        # return that best value that we've found
        return best_move


    def max_value(self, node):
        print "MiniMax–>MAX: Visited Node :: " + node.Name
        if self.isTerminal(node):
            return self.getUtility(node)

        infinity = float('inf')
        max_value = -infinity

        successors_states = self.getSuccessors(node)
        for state in successors_states:
            max_value = max(max_value, self.min_value(state))
        return max_value

    def min_value(self, node):
        print "MiniMax–>MIN: Visited Node :: " + node.Name
        if self.isTerminal(node):
            return self.getUtility(node)

        infinity = float('inf')
        min_value = infinity

        successor_states = self.getSuccessors(node)
        for state in successor_states:
            min_value = min(min_value, self.max_value(state))
        return min_value

     #   UTILITY METHODS   #
    # successor states in a game tree are the child nodes…
    def getSuccessors(self, node):
        assert node is not None
        return node.children

    # return true if the node has NO children (successor states)
    # return false if the node has children (successor states)
    def isTerminal(self, node):
        assert node is not None
        return len(node.children) == 0

    def getUtility(self, node):
        assert node is not None
        return node.value

class AlphaBeta:
    # print utility value of root node (assuming it is max)
    # print names of all nodes visited during search
    def __init__(self, game_tree):
        self.game_tree = game_tree  # GameTree
        self.root = game_tree.root  # GameNode
        return

    def alpha_beta_search(self, node):
        infinity = float('inf')
        best_val = -infinity
        beta = infinity

        successors = self.getSuccessors(node)
        best_state = None
        for state in successors:
            value = self.min_value(state, best_val, beta)
            if value > best_val:
                best_val = value
                best_state = state
        print "AlphaBeta:  Utility Value of Root Node: = " + str(best_val)
        print "AlphaBeta:  Best State is: " + best_state.Name
        return best_state

    def max_value(self, node, alpha, beta):
        print "AlphaBeta–>MAX: Visited Node :: " + node.Name
        if self.isTerminal(node):
            return self.getUtility(node)
        infinity = float('inf')
        value = -infinity

        successors = self.getSuccessors(node)
        for state in successors:
            value = max(value, self.min_value(state, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, node, alpha, beta):
        print "AlphaBeta–>MIN: Visited Node :: " + node.Name
        if self.isTerminal(node):
            return self.getUtility(node)
        infinity = float('inf')
        value = infinity

        successors = self.getSuccessors(node)
        for state in successors:
            value = min(value, self.max_value(state, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)

        return value
    #                     #
    #   UTILITY METHODS   #
    #                     #

    # successor states in a game tree are the child nodes…
    def getSuccessors(self, node):
        assert node is not None
        return node.children

    # return true if the node has NO children (successor states)
    # return false if the node has children (successor states)
    def isTerminal(self, node):
        assert node is not None
        return len(node.children) == 0

    def getUtility(self, node):
        assert node is not None
        return node.value
==================================
DFS 
# Using a Python dictionary to act as an adjacency list
graph = {
  '5' : ['3','7'],
  '3' : ['2', '4'],
  '7' : ['8'],
  '2' : [],
  '4' : ['8'],
  '8' : []
}

visited = set() # Set to keep track of visited nodes of graph.

def dfs(visited, graph, node):  #function for dfs 
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

# Driver Code
print("Following is the Depth-First Search")
dfs(visited, graph, '5')
================================
DFS-ID
# Python program to print DFS traversal from a given
# given graph
from collections import defaultdict
 
# This class represents a directed graph using adjacency
# list representation
class Graph:
 
    def __init__(self,vertices):
 
        # No. of vertices
        self.V = vertices
 
        # default dictionary to store graph
        self.graph = defaultdict(list)
 
    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)
 
    # A function to perform a Depth-Limited search
    # from given source 'src'
    def DLS(self,src,target,maxDepth):
 
        if src == target : return True
 
        # If reached the maximum depth, stop recursing.
        if maxDepth <= 0 : return False
 
        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[src]:
                if(self.DLS(i,target,maxDepth-1)):
                    return True
        return False
 
    # IDDFS to search if target is reachable from v.
    # It uses recursive DLS()
    def IDDFS(self,src, target, maxDepth):
 
        # Repeatedly depth-limit search till the
        # maximum depth
        for i in range(maxDepth):
            if (self.DLS(src, target, i)):
                return True
        return False
 
# Create a graph given in the above diagram
g = Graph (7);
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 3)
g.addEdge(1, 4)
g.addEdge(2, 5)
g.addEdge(2, 6)
 
target = 6; maxDepth = 3; src = 0
 
if g.IDDFS(src, target, maxDepth) == True:
    print ("Target is reachable from source " +
        "within max depth")
else :
    print ("Target is NOT reachable from source " +
        "within max depth")

Steepest hill climbing 
import random

# Define your objective function here, you can replace this with your own function
def objective_function(x):
    return -(x ** 2)  # Example function to minimize x^2

# Steepest Hill Climb Algorithm
def steepest_hill_climb(max_iterations, step_size, initial_solution):
    current_solution = initial_solution
    current_value = objective_function(current_solution)

    for _ in range(max_iterations):
        neighbor_solutions = [current_solution + step_size, current_solution - step_size]
        neighbor_values = [objective_function(neighbor) for neighbor in neighbor_solutions]
        best_neighbor_value = max(neighbor_values)
        
        if best_neighbor_value <= current_value:
            break
        
        best_neighbor_index = neighbor_values.index(best_neighbor_value)
        current_solution = neighbor_solutions[best_neighbor_index]
        current_value = best_neighbor_value

    return current_solution, current_value

# Example usage
max_iterations = 1000
step_size = 0.1
initial_solution = random.uniform(-10, 10)

best_solution, best_value = steepest_hill_climb(max_iterations, step_size, initial_solution)
print("Best Solution:", best_solution)
print("Objective Value at Best Solution:", best_value)=========

Dfs id
# Depth-First Search (DFS) with Iterative Deepening (ID) in Python

# Graph represented as an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# Depth-First Search (DFS) function
def dfs(graph, start, goal, max_depth):
    visited = set()
    stack = [(start, 0)]

    while stack:
        node, depth = stack.pop()
        visited.add(node)
        
        if node == goal:
            return True
        
        if depth < max_depth:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append((neighbor, depth + 1))
    
    return False

# Iterative Deepening (ID) function
def iterative_deepening_dfs(graph, start, goal):
    max_depth = 0
    while True:
        if dfs(graph, start, goal, max_depth):
            return True
        max_depth += 1

    return False

# Example usage
start_node = 'A'
goal_node = 'F'
if iterative_deepening_dfs(graph, start_node, goal_node):
    print("Goal node", goal_node, "found from node", start_node)
else:
    print("Goal node", goal_node, "not found from node", start_node)