routes = []
persons = ['A', 'B', 'C', 'D', 'E']  #参与传球的人
N = 5  #传球次数
first_person = 'A'  # 第一个传球出去的人
last_person = 'A'  # 最后一个接球的人

print '第一个传球的人是：%s\n最后一个接球的人是：%s\n传球次数：%d' % (first_person, last_person, N)

# 传球之前
routes.append([first_person])

# 第1次至N次传球
for i in range(1, N+1):
    routes.append([])    
    for route in routes[i-1]:
        for person in persons:
            if i == N and person != last_person:
                pass
            elif route[-1] == person:
                pass
            elif route[-1] == 'A' and person == 'B':
                pass
            elif route[-1] == 'B' and person == 'A':
                pass
            elif route[-1] == 'C' and person != 'D':
                pass
            elif route[-1] == 'E' and person == 'C':
                pass
            else:                
                routes[i].append(route + person)
    print '第%d次传球后总共有%d条路径：' % (i, len(routes[i]))
    print routes[i]
