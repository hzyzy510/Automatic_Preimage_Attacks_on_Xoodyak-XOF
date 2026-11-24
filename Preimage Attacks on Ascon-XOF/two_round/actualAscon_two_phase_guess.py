import gurobipy as gp
from gurobipy import GRB

def variable_1(m,R):
    X = []
    for r in range(R):
        X.append(m.addVar(vtype=GRB.BINARY))
    return X
def check_one(x1,y2,expression):
    t3=[]#可能出现的变量
    for i in x1:
        t3.append(i)

    t1 = expression.split("+")

    t2 = []#0输出表达式中的变量
    for i in t1:
        for j in range(64):
            if "[" + str(j) + "][0]" in i:
                if j not in t2:
                    t2.append(j)

    x_in_y=[]#y2输出中有的变量
    for i in y2:
        y_variable=[]
        for j in t2:
            y_variable.append((i+j)%64)
        x_in_y.append(y_variable)
    result=[]
    for i in range(len(x_in_y)):
        result_i=[]
        result_i.append("var"+str(y2[i]))
        for j in x_in_y[i]:
            if j not in t3:
                result_i.append(j)
        result.append(result_i)

    print(result)
def check_free(x1,y2,expression):
    t3=[]
    for i in range(64):
        if i not in x1:
            t3.append(i)

    t1 = expression.split("+")

    t2 = []
    for i in t1:
        for j in range(64):
            if "[" + str(j) + "][0]" in i:
                if j not in t2:
                    t2.append(j)

    x_in_y=[]
    for i in y2:
        for j in t2:
            if (i+j)%64 not in x_in_y:
                x_in_y.append((i+j)%64)

    for i in t3:
        if i in x_in_y:
            continue
        else:
            print(i,"不在线性方程中")
    print(len(x_in_y))
def get_x(expression):

    t1 = expression.split("+")
    t2 = []#项
    for i in t1:
        t2.append(i)
    x_anf = []  # 每一项对应的变量
    for i in t2:
        for x in range(64):
            if "[" + str(x) + "][0]" in i:
                if x not in x_anf:
                    x_anf.append(x)

    return x_anf
def get_x_time(expression):

    t1 = expression.split("+")
    t2 = []#大于一次的项
    for i in t1:
        if "*" in i:
            if count_char(i, "x") >= 1:
                t2.append(i)

    x_time = []  # 每一项对应的变量
    for i in t2:
        t = []
        for x in range(64):
            if "[" + str(x) + "][0]" in i:
                t.append([x, 1])
        x_time.append(t)
    return x_time

def count_char(string, char):
    count = 0
    for c in string:
        if c == char:
            count += 1
    return count

with open('actual_output.txt', 'r', encoding='utf-8') as file:
    data=file.read()

expressions = data.strip().split('\n')
x_anf_0=[]
x_time=[]
x_anf=[]
for i in range(len(expressions)):
    x_time.append(get_x_time(expressions[i]))
    x_anf.append(get_x(expressions[i]))
print(x_anf)
print(x_time[0])
result=[]
count_n2=34
m=gp.Model("milp12")
X=variable_1(m,64)
Y_multiple=variable_1(m, 64)#一开始被线性化的部分
X_multiple=variable_1(m, 64)#新的被线性求解求出来的变量加入线性化
Y=variable_1(m, 64)#最终被线性化的部分
for i in range(64):
    m.addConstr(X_multiple[i] + X[i] <= 1)
    m.addConstr(Y[i] >= Y_multiple[i])

for x in range(64):
    m.addConstr(gp.quicksum([Y_multiple[(x - k + 64) % 64] for k in x_anf[0]]) >= X_multiple[x])
    m.addConstr(gp.quicksum([Y[(x - k + 64) % 64] for k in x_anf[0]]) >= 1 - X[x] - X_multiple[x])#保证输出中包含未被猜测的变量
    m.addConstr((gp.quicksum([X[(k + x) % 64] + X_multiple[(k + x) % 64] for k in x_anf[0]]) - (len(x_anf[0]))) <= len(x_anf[0]) * (1 - Y_multiple[x]))
    m.addConstr((gp.quicksum([X[(k + x) % 64] + X_multiple[(k + x) % 64] for k in x_anf[0]]) - (len(x_anf[0]))) >= -len(x_anf[0]) * (1 - Y_multiple[x]))
    m.addConstr(len(x_anf[0]) * Y_multiple[x] >= (gp.quicksum([X[(k + x) % 64] + X_multiple[(k + x) % 64] for k in x_anf[0]]) - (len(x_anf[0]) - 1)))
    for i in x_time[x]:
        tX=[]
        for j in i:
            tX.append(j[0])
        if len(tX)>=2:
            m.addConstr(gp.quicksum([X[k] for k in tX]) >= (len(tX) - 1) * Y_multiple[x])
            m.addConstr(gp.quicksum([X_multiple[k] + X[k] for k in tX]) >= (len(tX) - 1) * Y[x])
m.addConstr(gp.quicksum(X_multiple) == gp.quicksum(Y_multiple))
m.addConstr(gp.quicksum(Y) + gp.quicksum(X) == 63)

m.setObjective(gp.quicksum(X), GRB.MINIMIZE)

m.optimize()

result.append([sum([X[i].X for i in range(64)]), sum([Y[i].X for i in range(64)])])
print("the number of guessed variables：",sum([X[i].X for i in range(64)]))
print("the number of linarized ouput bits：", sum([Y[i].X for i in range(64)]))
print(result)

tX=[]
tY_multiple=[]
tX_multiple=[]
tY=[]
for x in range(64):
    if X[x].X==1:
        tX.append(x)
    if Y_multiple[x].X==1:
        tY_multiple.append(x)
    if X_multiple[x].X==1:
        tX_multiple.append(x)
    if Y[x].X==1:
        tY.append(x)

print("gussed_var：", tX)
print("linearized_output：", tY_multiple)
print("new_var：", tX_multiple)
print("final_linearized_output：", tY)

