# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:10:26 2017

@author: 呵呵
"""

import pandas as pd
import numpy as np
from pandas import Series,DataFrame


#---------------------------《利用python进行数据分析》--------------------------
#第五章，Pandas入门
#-----------------------------------------------------------------------------
#Series

#-----------------------------------------------------------------------------
#创建一个Series
obj = Series([4,5,1,2])

obj.values 

Out[11]: array([4, 5, 1, 2], dtype=int64)

obj.index
RangeIndex(start=0, stop=4, step=1)


#创建一个字母索引的Series
obj2 = Series([1,2,3,4,1],index  = ['a','b','c','d','e'])

obj2['a']
Out[18]: 1

obj2[['a','b','d']]
Out[20]: 
a    1
b    2
d    4
dtype: int64

#-----------------------------------------------------------------------------
obj2[obj2>2]        #选取obj》2的部分并在obj中显示
Out[26]: 
c    3
d    4
dtype: int64

obj2*2
np.exp(obj2)        #计算e的次方
pd.isnull(obj)      #检测缺失数据
pd.isnotnull(obj)   #检测缺失数据
obj.isnull()        #检测缺失数据
obj.name = 'XXXX'  #给Series的值设置一个名称
obj.index.name = 'XXXX'  #给Series的索引设置一个名称

obj.index = ['a','b','c','d'] #修改Series的索引


#-----------------------------------------------------------------------------

#DataFram


data = {'statw':['owio','owio','owio','haha','haha'],
        'year' :[2000,2001,2002,2003,2004],
        'pop'  :[1.5,1.7,1.3,1.3,1.9]
        }

data = pd.DataFrame(data)

DataFrame(data,columns = ['year','statw','pop']) #设置列序列
DataFrame(data,columns = ['year','hehehe','statw','pop','hahaha'])  # 新增两个列序列
data.columns #显示数据的列序列名称
data.ix['xx'] data.ix[0:10] #取出一行或几行行的data数据
data['XX'] = 10 #整列赋值为10
data['year'] = np.arange(5.) #为列赋值 0-4 
data['esten'] = data.statw =='XXX' #新增一列
del data['esten']

pop = {'nevada':{2001:2.4,2002:2.9},
       'ohio':{2000:1.5,2001:1.7,2002:3.6}
       }
       #外层字典为列，内层键作为行索引
       
frame3 = DataFrame(pop)
frame3 = DataFrame(pop,index = ['1999','2000','2001','2002','2003','2004','2005'])
#改变索引会导致值全部变为nan
data[:] = np.random.randn(5,3)

#-----------------------------------------------------------------------------
#基本功能：
#-----------------------------------------------------------------------------
#重新索引：

obj= Series([1,2,3,4,1],index  = ['a','b','c','d','e'])
obj2 = obj.reindex(index = ['b','A','d','d','e','f','f'])
#reindex后，索引未变的数据会保留并随索引排序，变化的索引下的数据会设置为nan
obj3 = Series(['blue','green','yellow'],index = [0,2,4])
obj3.reindex(index = [np.arange(5)],method='ffill') #ffill pad 从上往下填充
obj3.reindex(index = [np.arange(5)],method='bfill') #bfill backfill 从上往下填充

frame = DataFrame(np.random.randint(1,10,9).reshape(3,3),index=['a','c','d'],columns  = ['ohio','texas','cali'])
frame2 = frame.reindex(['a','b','c','d'])
frame2 =frame2.reindex(columns = states)

frame2 = frame2.ix[['a','c','b','e'],states]

"""
总结:
method , 差值的填充方式
fill_value 重新索引时候填入缺失值使用的替代值
limit, 前向或后向填充时最大填充量
level 在MultiIndex指定级别上匹配简单索引，否则选取子集
copy 默认True，无论如何都复制，如果False，新旧相等就不复制。
"""

#-----------------------------------------------------------------------------
#丢弃指定轴上的项目
obj = Series(np.arange(5),index = ['a','b','c','d','e'])
new_obj  = obj.drop('c')

frame = DataFrame(np.random.randint(1,10,9).reshape(3,3),index=['haha','hehe','huhu'],columns=['a','b','c'])
frame.drop('haha') #丢弃行
frame.drop('a',axis=1) #丢弃列
#-----------------------------------------------------------------------------
#索引，选取，过滤
obj['b']
obj[1]
obj[2:4]
obj[['b','a','d']]
obj[[1,3]] #有点不懂
obj[obj<2] 
obj['a':'c']
obj['a':'c'] = 2


data = DataFrame(np.random.randint(1,10,16).reshape(4,4),index = ['ohio','colorad','haha','hehe'],columns=['one','two','three','four'])

data['two']
data[['three','two']]
data[:2]
data[data['three']>7]
data['one'][data['three']>7] = 4
#找到这个关键语句了，得知一系列标签的值，
frame['a'][frame['b'].isnull()]
#找到这个关键语句了，得知一系列标签的值，  

data[data<5]= 0

data[:]  = np.random.randint(1,100,16).reshape(4,4)
#可以对数据进行重新填充

#-----------------------------------------------------------------------------
#算数运算和数据对齐
s1 = Series(np.random.randn(4),index=list('abcd'))
s2 = Series(np.random.randn(5),index=list('acedf'))
s1 + s2
s1-s2
s1*s2

df1 = DataFrame(np.random.randint(1,100,9).reshape(3,3),columns=list('bcd'),index = ['ohio','texas','colorado'])
df2 = DataFrame(np.random.randint(1,100,12).reshape(4,3),columns=list('bde'),index = ['Utah','ohio','texas','oregen'])

s1 + s2
s1-s2
s1*s2

df1+df2 
df1.add(df2,fill_value=0) 
#关键在于fill_value=0 ，相当于df1+df2，不可相加的位置，由各自的值分别填充
df1.sub(df2,fill_value=0) #减法
df1.div(df2,fill_value=0) #除法
df1.mul(df2,fill_value=0) #乘法

df1.reindex(columns=list('bde'),fill_value= 0)
#重新索引时也可用fill_value
#-----------------------------------------------------------------------------
#dataframe与Series间的运算


arr = np.arange(12).reshape(3,4)
arr-arr[0]
Out[335]: 
array([[0, 0, 0, 0],
       [4, 4, 4, 4],
       [8, 8, 8, 8]])

#广播运算

df1 = DataFrame(np.random.randint(1,100,9).reshape(3,3),columns=list('bcd'),index = ['ohio','texas','colorado'])

ser1 = df1.iloc[1]

ser1 = df1.iloc[1]

df1- ser1
Out[357]: 
           b   c   d
ohio     -15  48  12
texas      0   0   0
colorado -16  -6 -39


#-----------------------------------------------------------------------------
#函数应用和运算
frame = DataFrame(np.random.randint(1,100,12).reshape(4,3),columns=list('bde'),index = ['utah','phip','texas','oregen'])
np.abs(frame)

f = lambda x : x.max() - x.min()
frame.apply(f,axis=0)


def f(x):
    return Series([x.min(),x.max()],index = ['min','max'])


format1 = lambda x: '%.2f' % x  #编程小数点后面两位的浮点
frame.applymap(format1)
frame['e'].map(format1)

"""
applymap:
        Apply a function to a DataFrame that is intended to operate
        elementwise, i.e. like doing map(func, series) for each series in the
        DataFrame
"""

#-----------------------------------------------------------------------------
#排序和排名

frame = DataFrame(np.random.randint(1,100,12).reshape(4,3),columns=list('bda'),index = ['utah','phip','texas','oregen'])
frame.sort_index(1)
frame.sort_index(0)

obj = Series([4,77,-34,1,2,4,234])
obj.order

frame.sort_values(by='d')

obj.rank()
frame.rank(method='average')

"""
method说明
'average' 在相等分组中，为各个值分配平均排名
'min' 使用整个分组的最小排名
'max' 使用整个分组的最大排名
'first' 按值在原始数据中的出现顺序分配排名
"""

#-----------------------------------------------------------------------------
#汇总和计算描述统计
frame.sum(axis=1)

frame.mean(axis=1,skipna=False) #有na项的不会返回值了

"""
axis 轴
skipna 排除缺失值，默认为True，有na项的不会返回值了
level 用于层次化索引的，根据level进行分组简约
"""
frame = DataFrame(np.random.randint(1,100,12).reshape(4,3),columns=list('bda'),index = ['utah','phip','texas','oregen'])

frame.cunsum(axis = 1) #累加
frame.cumsum()
Out[574]: 
          b    d    a
utah     84   81   76
phip    100  115  116
texas   174  142  161
oregen  241  180  193


frame.idxmax(axis=1) #返回值最大的标签名 可以横轴也可以纵轴
frame.idxmin()#返回值最小的标签名 可以横轴也可以纵轴

frame.describe()

"""

描述和汇总统计
count 计算非na的值
describe 汇总统计
min , max 计算最大最小值
好像没了------argmin,argmax 计算能够获取到最小最大值得索引位置
idxmin,idxmax 计算能够获取到最小最大值得标签名
frame.quantile() 计算样本分位数
sum
mean
median 算数中位数（50%）
mad 根据平均值计算平均绝对离差
var 方差
std 标准差
skew 偏度
kurt 峰度
cumsum 累计和
cummax cummin 累计最大最小值
cumprod 累计积
diff 计算一阶差分
pct_change 计算百分数变化

"""


#-----------------------------------------------------------------------------
#相关系数与协方差
#略

#-----------------------------------------------------------------------------
#唯一值，值计数，成员资格方法

obj.unique() #返回唯一值
obj.value_counts #计算各值出现的频率
pd.value_counts(obj)#计算各值出现的频率

mask = obj.isin(['b','c']) 返回bool值
obj[mask] 


"""
isin 计算一个‘表示Series各值是否包含于传入的值序列中’的bool数组
unique 计算Series中的唯一值，按发现的顺序返回
value_counts 返回一个Series，其索引为唯一值，值为频率，按计数值降序排列
"""

#-----------------------------------------------------------------------------
#处理缺失数据

obj = Series(['sddsg','svdq','ssaca',np.nan,'wqqwc','asd','qweqw'])
obj.isnull() 

"""
dropna() 丢掉存在缺失数据的行或列
fillna() 用指定值或插值方法（ffill,bfill）填充缺失数据
isnull() 返回bool对象，这些布尔值表示那些事缺失的，
notnull() 与上面相反
""" 

#-----------------------------------------------------------------------------
#处理缺失数据

data.dropna(axis=1,how='all') #丢失一行或一列所有都为na的数据，默认为行
data.dropna(thresh=2) #没懂什么意思



#-----------------------------------------------------------------------------
#填充缺失数据
data.fillna(0)

data.fillna({"d":0.5,"a":-1}) #列columns标签下的na数据进行填充
data.fillna(0,inplace = True)
data.fillna(method= 'ffill')
data.fillna(method="bfill",axis =1)
data.fillna(data.mean())


#-----------------------------------------------------------------------------
#层次化索引

data = Series(np.random.randn(12),index = [list('aaabbbcccddd'),np.arange(12)])
data['a':'b'] #根据第一层标签选择数据
data[:,1] # 根据第二层标签选择数据
data.unstack() #解开层次化索引
data1.stack()  #行列转变为层次化索引

frame.columns.names = ['state'] #给列标签命名
frame.index.names = ['key1','key2'] #给索引命名



frame.swaplevel().sortlevel() #交换索引的位置
"""
frame
Out[995]: 
        utah  phip  texas  oregen
haha a    92     5      2      92
hehe b    95    88     71      39
hoho c    18    89     43      19

frame.swaplevel().sortlevel()
Out[994]: 
        utah  phip  texas  oregen
a haha    92     5      2      92
b hehe    95    88     71      39
c hoho    18    89     43      19

"""
#-----------------------------------------------------------------------------
#根据级别汇总统计
frame = DataFrame(np.random.randint(1,100,12).reshape(3,4),
                  index = [['ha','ha','he'],['a','b','c']],
                  columns = [['state','state','country','country'],['uteah','phip','texas','oregen']])

frame.columns.names = ['sta','ca'] #给列标签命名
frame.index.names = ['key1','key2'] #给索引命名


frame.sum(level='key1')
frame.sum(level = ['sta'] ,axis=1) #相当于用了groupby



#-----------------------------------------------------------------------------
#使用DATAFRAM的列





frame = DataFrame({'a':range(7),
                   'b':range(7,0,-1),
                   'c': ['one','one','one','two','two','two','two'],
                   'd': [0,1,2,1,2,0,1]
                   })

frame2 = frame.set_index(['c','d'])
frame.set_index(['c','d'],drop=False)
frame2.reset_index()




#-----------------------------------------------------------------------------
#其他
a = Series(np.arange(5.))
非整数索引


#---------------------------《利用python进行数据分析》--------------------------
#第七章 数据规整化：清理、转换、合并、重塑
#-----------------------------------------------------------------------------

#合并数据集
#-----------------------------------------------------------------------------

df1 = DataFrame({'key':['b','b','a','c','a','a','b'],
                 'data1':np.arange(7)
                 })

df2 = DataFrame({'key':['a','b','d'],
                 'data2':np.arange(3)
                 })



pd.merge(df1,df2) #不指定的话会将重叠列的列名当做键
pd.merge(df1,df2,on='key') # 默认情况下做的inner连接

pd.merge(df1,df2,how='inner') #默认取两个组的交集
pd.merge(df1,df2,how='outer') #取两个集合的并集
pd.merge(df1,df2,how='left')
pd.merge(df1,df2,how='right')



df3 = DataFrame({'1key':['b','b','a','c','a','a','b'],
                 'data1':np.arange(7)
                 })

df4 = DataFrame({'2key':['a','b','d'],
                 'data2':np.arange(3)
                 })

#两个数据集的值不同时的合并方法
pd.merge(df3,df4,left_on='1key',right_on='2key',how='outer')





left = DataFrame({'key1':['foo','foo','bar'],
                  'key2':['one','two','one'],
                  'lval':[1,2,3]})

right = DataFrame({'key1':['foo','foo','bar','bar'],
                   'key2':['one','one','one','two'],
                   'rval':[4,5,6,7]})

"""
left
Out[125]: 
  key1 key2  lval
0  foo  one     1
1  foo  two     2
2  bar  one     3

right
Out[126]: 
  key1 key2  rval
0  foo  one     4
1  foo  one     5
2  bar  one     6
3  bar  two     7
"""


 pd.merge(left,right,on=['key1','key2'],how = 'outer')
 
 
"""
  key1 key2  lval  rval
0  foo  one   1.0   4.0
1  foo  one   1.0   5.0
2  foo  two   2.0   NaN
3  bar  one   3.0   6.0
4  bar  two   NaN   7.0
"""




#索引上的合并
#-----------------------------------------------------------------------------

left1 = DataFrame({'key':['a','b','a','a','b','c'],
              'value':range(6)})
right1 = DataFrame({'group_val':[3.5,7]}, index=['a','b'])


"""
left1
Out[133]: 
  key  value
0   a      0
1   b      1
2   a      2
3   a      3
4   b      4
5   c      5

right1
Out[134]: 
   group_val
a        3.5
b        7.0


"""
pd.merge(left1,right1,left_on='key',right_index='True',how='outer')
"""
Out[163]: 
  key  value  group_val
0   a      0        3.5
2   a      2        3.5
3   a      3        3.5
1   b      1        7.0
4   b      4        7.0
5   c      5        NaN
"""



lefth = DataFrame({'key1':['ohio','ohio','ohio','nevada','nevada'],
                   'key2':[2000,2001,2002,2001,2002],
                   'data':np.arange(5.)})



righth = DataFrame(np.arange(12).reshape(6,2),
                   index = [['nevada','nevada','ohio','ohio','ohio','ohio'],
                   [2001,2000,2000,2000,2001,2002]],
                   columns=  ['event1','event2'])
"""
lefth
Out[159]: 
   data    key1  key2
0   0.0    ohio  2000
1   1.0    ohio  2001
2   2.0    ohio  2002
3   3.0  nevada  2001
4   4.0  nevada  2002

righth
Out[160]: 
             event1  event2
nevada 2001       0       1
       2000       2       3
ohio   2000       4       5
       2000       6       7
       2001       8       9
       2002      10      11
"""


pd.merge(lefth,righth,left_on=['key1','key2'], right_index = True,how = 'outer')
"""
Out[162]: 
   data    key1    key2  event1  event2
0   0.0    ohio  2000.0     4.0     5.0
0   0.0    ohio  2000.0     6.0     7.0
1   1.0    ohio  2001.0     8.0     9.0
2   2.0    ohio  2002.0    10.0    11.0
3   3.0  nevada  2001.0     0.0     1.0
4   4.0  nevada  2002.0     NaN     NaN
4   NaN  nevada  2000.0     2.0     3.0

"""

left2 = DataFrame(np.arange(1.,7.).reshape(3,2),
                  index = list('ace'),
                  columns = ['ohio','nevada'])
                  
right2 = DataFrame(np.arange(7.,15.).reshape(4,2),
                  index = list('bcde'),
                  columns = ['missouri','albana'])

"""
left2
Out[178]: 
   ohio  nevada
a   1.0     2.0
c   3.0     4.0
e   5.0     6.0

right2
Out[179]: 
   missouri  albana
b       7.0     8.0
c       9.0    10.0
d      11.0    12.0
e      13.0    14.0
"""

pd.merge(left2,right2,how='outer',left_index=True,right_index=True)

"""
Out[181]: 
   ohio  nevada  missouri  albana
a   1.0     2.0       NaN     NaN
b   NaN     NaN       7.0     8.0
c   3.0     4.0       9.0    10.0
d   NaN     NaN      11.0    12.0
e   5.0     6.0      13.0    14.0

"""

left2.join(right2,how='outer')

"""
Out[181]: 
   ohio  nevada  missouri  albana
a   1.0     2.0       NaN     NaN
b   NaN     NaN       7.0     8.0
c   3.0     4.0       9.0    10.0
d   NaN     NaN      11.0    12.0
e   5.0     6.0      13.0    14.0

"""
another = DataFrame([[7,8],[9,10],[11,12],[16,17]],
                    index =list('acef'),
                    columns = ['new york','oregen'])
"""

Out[192]: 
   new york  oregen
a         7       8
c         9      10
e        11      12
f        16      17

"""

left2.join(right2,another,how='outer')

"""
Out[198]: 
   ohio  nevada  missouri  albana  new york  oregen
a   1.0     2.0       NaN     NaN       7.0     8.0
b   NaN     NaN       7.0     8.0       NaN     NaN
c   3.0     4.0       9.0    10.0       9.0    10.0
d   NaN     NaN      11.0    12.0       NaN     NaN
e   5.0     6.0      13.0    14.0      11.0    12.0
f   NaN     NaN       NaN     NaN      16.0    17.0
"""



#轴向连接
#-----------------------------------------------------------------------------
arr = np.arange(12).reshape(3,4)

"""
Out[206]: 
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
"""

np.concatenate([arr,arr],axis = 1)

"""
Out[205]: 
array([[ 0,  1,  2, ...,  1,  2,  3],
       [ 4,  5,  6, ...,  5,  6,  7],
       [ 8,  9, 10, ...,  9, 10, 11]])

"""

s1 = Series([0,1],index = ['a','b'])
s2 = Series([2,3,4],index = ['c','d','e'])
s3 = Series([5,6],index = ['f','g'])

pd.concat([s1,s2,s3])

"""
Out[233]: 
a    0
b    1
c    2
d    3
e    4
f    5
g    6
dtype: int64
"""

pd.concat([s1,s2,s3],axis=1)

"""
Out[234]: 
     0    1    2
a  0.0  NaN  NaN
b  1.0  NaN  NaN
c  NaN  2.0  NaN
d  NaN  3.0  NaN
e  NaN  4.0  NaN
f  NaN  NaN  5.0
g  NaN  NaN  6.0

"""

s4 = pd.concat([s1*5,s3])
"""
Out[241]: 
a    0
b    5
f    5
g    6
"""
pd.concat([s1,s4],axis=1)
"""
Out[251]: 
     0  1
a  0.0  0
b  1.0  5
f  NaN  5
g  NaN  6
"""

pd.concat([s1,s4],axis=1,join='inner')
"""
Out[250]: 
   0  1
a  0  0
b  1  5
"""

#合并的过程，为每个合并的单元，创建层次化索引：

result = pd.concat([s1,s2,s3],keys=['one','two','three'])

"""
Out[253]: 
one    a    0
       b    1
two    c    2
       d    3
       e    4
three  f    5
       g    6
dtype: int64

"""


result = pd.concat([s1,s2,s3],keys=['one','two','three'],axis= 1)

"""
Out[256]: 
   one  two  three
a  0.0  NaN    NaN
b  1.0  NaN    NaN
c  NaN  2.0    NaN
d  NaN  3.0    NaN
e  NaN  4.0    NaN
f  NaN  NaN    5.0
g  NaN  NaN    6.0
"""

df1 = DataFrame(np.arange(6).reshape(3,2),
                index = list('abc'),
                columns = ['one','two'])

df2 = DataFrame(5+np.arange(4).reshape(2,2),
                index = list('ac'),
                columns = ['three','four'])

"""
df1
Out[273]: 
   one  two
a    0    1
b    2    3
c    4    5

df2
Out[274]: 
   three  four
a      5     6
c      7     8
"""

pd.concat([df1,df2],axis=1,keys=['l1','l2'])

"""
Out[268]: 
   l1        l2     
  one two three four
a   0   1   5.0  6.0
b   2   3   NaN  NaN
c   4   5   7.0  8.0
"""

pd.concat([df1,df2])

"""
Out[269]: 
   four  one  three  two
a   NaN  0.0    NaN  1.0
b   NaN  2.0    NaN  3.0
c   NaN  4.0    NaN  5.0
a   6.0  NaN    5.0  NaN
c   8.0  NaN    7.0  NaN

"""
pd.concat({'l1':df1,'l2':df2},axis=1)
 
"""
 Out[275]: 
   l1        l2     
  one two three four
a   0   1   5.0  6.0
b   2   3   NaN  NaN
c   4   5   7.0  8.0
"""


df1 = DataFrame(np.random.randn(3,4),columns= list('abcd'))
df2 = DataFrame(np.random.randn(2,3),columns= list('bda'))

pd.concat([df1,df2],ignore_index=True)

""""
Out[291]: 
          a         b         c         d
0 -0.137209  0.016283 -0.308824 -1.300364
1 -1.476067  0.513266 -2.130261 -3.258791
2  0.107244 -3.029398 -2.149923 -0.044807
3 -2.091144 -1.110878       NaN  1.529865
4  0.318508 -1.049669       NaN  0.672157
"""

pd.concat([df1,df2])

"""
Out[292]: 
          a         b         c         d
0 -0.137209  0.016283 -0.308824 -1.300364
1 -1.476067  0.513266 -2.130261 -3.258791
2  0.107244 -3.029398 -2.149923 -0.044807
0 -2.091144 -1.110878       NaN  1.529865
1  0.318508 -1.049669       NaN  0.672157

"""


"""

objs 参与连接的pandas对象的列表或字典，唯一必须的参数
axis  轴
join inner,outer 交集 并集
join_axes 指明用于其他n-1条轴的索引，不执行并集/交集运算
keys  与连接对象有关的关键值
level 指定用于层次化索引各级别上的索引，如果设置的keys的话
names 创建分层级别的名称，如果设置了keys或levels的话
verify_integrity 检查二级果对象新轴上的重复情况，如果发现则引发异常，默认允许
ignore_index 不保留连接轴上的US噢因，产生一组新索引


"""




#合并重叠数据
#-----------------------------------------------------------------------------


a = Series([np.nan,2.5,np.nan,3.5,4.5,np.nan],index = ['f','e','d','c','b','a'])
b = Series(np.arange(len(a)),index = ['f','e','d','c','b','a'],dtype = np.float64)
b[-1] = np.nan


"""
a
Out[295]: 
f    NaN
e    2.5
d    NaN
c    3.5
b    4.5
a    NaN
dtype: float64

b
Out[5]: 
f    0.0
e    1.0
d    2.0
c    3.0
b    4.0
a    NaN
dtype: float64
"""

np.where(pd.isnull(a),b,a) #搞毛啊？这是？


"""
array([ 0. ,  2.5,  2. ,  3.5,  4.5,  nan])
"""


b.combine_first(a) #以B为优先值，空缺由a填补

"""
Out[23]: 
f    0.0
e    1.0
d    2.0
c    3.0
b    4.0
a    NaN
dtype: float64
"""

a.combine_first(b)   #以a为优先值，空缺由b填补

"""
Out[24]: 
f    0.0
e    2.5
d    2.0
c    3.5
b    4.5
a    NaN
dtype: float64
"""

df1 = DataFrame({'a':[1.,np.nan,5.,np.nan],
                 'b':[np.nan,2.,np.nan,6.],
                 'c':range(2,18,4) 
                 })

df2 = DataFrame({'a':[5.,4.,np.nan,3.,7.],
                 'b':[np.nan,3.,4.,6.,8.]
                 })


"""
df1
Out[43]: 
     a    b   c
0  1.0  NaN   2
1  NaN  2.0   6
2  5.0  NaN  10
3  NaN  6.0  14

df2
Out[44]: 
     a    b
0  5.0  NaN
1  4.0  3.0
2  NaN  4.0
3  3.0  6.0
4  7.0  8.0
"""
df1.combine_first(df2)

"""
Out[46]: 
     a    b     c
0  1.0  NaN   2.0
1  4.0  2.0   6.0
2  5.0  4.0  10.0
3  3.0  6.0  14.0
4  7.0  8.0   NaN

"""






#重塑和轴向旋转
#-----------------------------------------------------------------------------

#重塑层次化索引


data = DataFrame(np.arange(6).reshape(2,3),
                 index = ['ohio','colorado'],
                 columns= ['one','two','three'])

data.index.name = 'states'
data.columns.name = 'number'

re = data.stack()

re.unstack('states')
re.unstack(0)
# 操作的均是最内层。

"""
states    number
ohio      one       0
          two       1
          three     2
colorado  one       3
          two       4
          three     5
"""



s1 = Series([0,1,2,3],index=list('abcd'))
s2 = Series([4,5,6],index=list('cde'))
data2 = pd.concat([s1,s2],keys=['one','two'],axis=1)

"""
Out[178]: 
one  a    0
     b    1
     c    2
     d    3
two  c    4
     d    5
     e    6
dtype: int64
"""
data2.unstack().stack()  #可逆的
data2.unstack().stack(dropna = False) #不丢掉na数据


df = DataFrame({'left':re,'right':re+5})
df.columns.name = 'side'

"""

Out[207]: 
                 left  right
states   number             
ohio     one        0      5
         two        1      6
         three      2      7
colorado one        3      8
         two        4      9
         three      5     10
         
"""





df.unstack(0)
"""
Out[230]: 
side   left          right         
states ohio colorado  ohio colorado
number                             
one       0        3     5        8
two       1        4     6        9
three     2        5     7       10
"""




df.unstack(1)
"""
Out[231]: 
side     left           right          
number    one two three   one two three
states                                 
ohio        0   1     2     5   6     7
colorado    3   4     5     8   9    10
"""

df.unstack('states')
"""
Out[246]: 
side   left          right         
states ohio colorado  ohio colorado
number                             
one       0        3     5        8
two       1        4     6        9
three     2        5     7       10
"""

df.unstack('states').stack('side')

"""
states        ohio  colorado
number side                 
one    left      0         3
       right     5         8
two    left      1         4
       right     6         9
three  left      2         5
       right     7        10

"""





#将‘长格式’旋转为‘宽格式’
#-----------------------------------------------------------------------------



#数据转换
#-----------------------------------------------------------------------------
#移除重复数据
data = DataFrame({'k1':['one']*3+['two']*4,
                  'k2':[1,1,2,3,3,4,4]
                  })



data[data.duplicated()]  #可以选择重复项
data = data.drop_duplicates() #去除重复项
#以上方法默认判断全部列 
data.drop_duplicates('k1')
data.drop_duplicates(['k1','k2'])
#选择判断哪些列




#-----------------------------------------------------------------------------
#利用函数或映射进行数据转换

#map函数

data =DataFrame({'food':['bacon','pulled pork','bacon','pastrami','corned beef','bacon','pastrami','honey ham','nova lox'],
                 'ounces':[4,3,12,6,7.5,8,3,5,6]
                 })


meat_to_animal  = {'bacon':'pig',
                   'pulled pork': 'pig',
                   'pastrami': 'cow',
                   'corned beef' : 'cow',
                   'honey ham':'pig',
                   'nova lox':'salmon'
                   }


data['animal'] = data['food'].map(str.lower).map(m)
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)


f = data['food'].map(lambda x: meat_to_animal[x.lower()])

"""
Out[443]: 

0       pig
1       pig
2       pig
3       cow
4       cow
5       pig
6       cow
7       pig
8    salmon
Name: food, dtype: object

"""


pd.merge(data,f,left_index=True,right_index=True)

"""
Out[447]: 
        food_x  ounces  food_y
                              
0        bacon     4.0     pig
1  pulled pork     3.0     pig
2        bacon    12.0     pig
3     pastrami     6.0     cow
4  corned beef     7.5     cow
5        bacon     8.0     pig
6     pastrami     3.0     cow
7    honey ham     5.0     pig
8     nova lox     6.0  salmon
"""


#-----------------------------------------------------------------------------
#替换值


data = Series([1.,-999.,2.,-999.,-1000.,3.])
data.replace(-999,np.nan)
data.replace([-999,-1000],np.nan)


data.replace([-999,-1000],[np.nan,0])  #将-999设置为nan,将-1000设置为0
data.replace({-999:np.nan,-1000:0}) #字典也可以实现




#-----------------------------------------------------------------------------
#重命名轴索引

data  = DataFrame(np.arange(12).reshape(3,4),
                  index = ['ohio','colorado','new york'],
                  columns = ['one','two','three','four'])

data.index = data.index.map(str.upper) #轴名大写
data.rename(index = str.title,columns= str.upper)

data.rename(index ={'OHIO':'haha'}) #用字典来替换index标签而不改变数值
data.rename(columns={'one':'haha'}) #用字典来替换col标签而不改变数值
data.rename(columns={'one':'haha'},inplace=True) #就地修改数据集

"""
          haha  two  three  four
haha        0    1      2     3
COLORADO    4    5      6     7
NEW YORK    8    9     10    11

"""



#-----------------------------------------------------------------------------
#离散化和面元划分

ages = [20,22,25,27,21,23,37,31,61,45,41,32]

bins  = [18,25,35,60,100]
cats = pd.cut(ages,bins)

"""
Out[54]: 
[(18, 25], (18, 25], (18, 25], (25, 35], (18, 25], ..., (25, 35], (60, 100], (35, 60], (35, 60], (25, 35]]
Length: 12
Categories (4, object): [(18, 25] < (25, 35] < (35, 60] < (60, 100]]

"""


cats.categories

"""
Out[57]: Index(['(18, 25]', '(25, 35]', '(35, 60]', '(60, 100]'], dtype='object')
"""

cats.codes
"""
Out[59]: array([0, 0, 0, ..., 2, 2, 1], dtype=int8)
"""
cats.value_counts()
pd.value_counts(cats)

"""
Out[71]: 
(18, 25]     5
(25, 35]     3
(35, 60]     3
(60, 100]    1
dtype: int64
"""


pd.cut(ages,[18,26,36,61,100],right=False)
#圆括号表示开端，方括号表示闭端（包括）

"""
Out[73]: 
[[18, 26), [18, 26), [18, 26), [26, 36), [18, 26), ..., [26, 36), [61, 100), [36, 61), [36, 61), [26, 36)]
Length: 12
Categories (4, object): [[18, 26) < [26, 36) < [36, 61) < [61, 100)]
"""

Gname = ['young','youngadult','middleaged','senior']

pd.cut(ages,bins,labels=Gname)
"""
[young, young, young, youngadult, young, ..., youngadult, senior, middleaged, middleaged, youngadult]
Length: 12
Categories (4, object): [young < youngadult < middleaged < senior]
"""


data = np.random.rand(1000)
cats = pd.qcut(data,4)  # 按4分位切割，每个档位的差相同，不懂pre的意思
cats.value_counts()

"""
[0.000949, 0.261]    250
(0.261, 0.515]       250
(0.515, 0.748]       250
(0.748, 1]           250
dtype: int64
"""

cats = pd.cut(data,4) #  按值4档位划分
cats.value_counts()
"""
Out[112]: 
(-1, 0.25]     242
(0.25, 0.5]    244
(0.5, 0.75]    266
(0.75, 1]      248
dtype: int64
"""

cats = pd.qcut(data,[0,0.1,0.5,0.9,1.])  #自己定义分位数，包含端点



               
#-----------------------------------------------------------------------------
#检测和过滤异常值

np.random.seed(12345)
data = DataFrame(np.random.randn(1000,4))

data.describe()

"""
                 0            1            2            3
count  1000.000000  1000.000000  1000.000000  1000.000000
mean     -0.067684     0.067924     0.025598    -0.002298
std       0.998035     0.992106     1.006835     0.996794
min      -3.428254    -3.548824    -3.184377    -3.745356
25%      -0.774890    -0.591841    -0.641675    -0.644144
50%      -0.116401     0.101143     0.002073    -0.013611
75%       0.616366     0.780282     0.680391     0.654328
max       3.366626     2.653656     3.260383     3.927528

"""

col = data[3]
col[np.abs(col)>3]
"""
Out[20]: 
97     3.927528
305   -3.399312
400   -3.745356
Name: 3, dtype: float64
"""

data[(np.abs(data)>1).any(1)] = np.sign(data)*3 
data.describe()
# np.sign 这个ufunc返回的是一个由1和-1组成的数组，表示原始值的符号
#any(1) 是什么鬼意思？


#-----------------------------------------------------------------------------
#排列和随机采样

df = DataFrame(np.arange(5*4).reshape(5,4))


"""
Out[101]: 
    0   1   2   3
0   0   1   2   3
1   4   5   6   7
2   8   9  10  11
3  12  13  14  15
4  16  17  18  19
"""
sampler = np.random.permutation(4)
"""
Out[100]: array([3, 0, 2, 1, 4])
"""
df.take(sampler)

"""

Out[102]: 
    0   1   2   3
3  12  13  14  15
0   0   1   2   3
2   8   9  10  11
1   4   5   6   7
4  16  17  18  19

"""
#相当于把sampler作为df的索引方式，
df.take(np.random.permutation(len(df)))

"""
Out[111]: 
    0   1   2   3
3  12  13  14  15
2   8   9  10  11
4  16  17  18  19
0   0   1   2   3
1   4   5   6   7

"""
bag = np.array([5,7,-1,6,4])
sample = np.random.randint(0,len(bag)-1,size=10)

"""
bag
Out[192]: array([ 5,  7, -1,  6,  4])

sample
Out[193]: array([0, 1, 0, 2, 0, 1, 0, 2, 1, 0])
"""

draws = bag.take(sample)
"""
draws
Out[195]: array([ 5,  7,  5, -1,  5,  7,  5, -1,  7,  5])

"""


#-----------------------------------------------------------------------------
#计算指标/哑变量


df = DataFrame({'key':['b','b','a','c','a','b'],
                'data1':range(6)})

pd.get_dummies(df['key'])

"""
Out[207]: 
     a    b    c
0  0.0  1.0  0.0
1  0.0  1.0  0.0
2  1.0  0.0  0.0
3  0.0  0.0  1.0
4  1.0  0.0  0.0
5  0.0  1.0  0.0

"""

dummies = pd.get_dummies(df['key'],prefix='key')

"""
Out[209]: 
   key_a  key_b  key_c
0    0.0    1.0    0.0
1    0.0    1.0    0.0
2    1.0    0.0    0.0
3    0.0    0.0    1.0
4    1.0    0.0    0.0
5    0.0    1.0    0.0
"""

df_with_dummy = df[['data1']].join(dummies)
"""
Out[239]: 
   data1  key_a  key_b  key_c
0      0    0.0    1.0    0.0
1      1    0.0    1.0    0.0
2      2    1.0    0.0    0.0
3      3    0.0    0.0    1.0
4      4    1.0    0.0    0.0
5      5    0.0    1.0    0.0
"""

#这个很有用，电影的案例没有原始数据 先不做了

value = np.random.rand(10)
bins=[0,0.2,0.4,0.6,0.8,1]
pd.get_dummies(pd.cut(value,bins))

"""

Out[268]: 
   (0, 0.2]  (0.2, 0.4]  (0.4, 0.6]  (0.6, 0.8]  (0.8, 1]
0       1.0         0.0         0.0         0.0       0.0
1       0.0         0.0         0.0         0.0       1.0
2       0.0         0.0         0.0         0.0       1.0
3       0.0         0.0         0.0         1.0       0.0
4       0.0         0.0         0.0         0.0       1.0
5       0.0         0.0         0.0         1.0       0.0
6       0.0         0.0         1.0         0.0       0.0
7       0.0         0.0         0.0         0.0       1.0
8       0.0         0.0         0.0         1.0       0.0
9       0.0         0.0         0.0         0.0       1.0

"""


#字符串操作
#-----------------------------------------------------------------------------



#-----------------------------------------------------------------------------
#字符串对象方法

val = 'a,b, guido'
val.split(',')


pieces = [x.strip() for x in val.split(',')]
pieces = val.strip().split(',')
"""
['a', 'b', ' guido']

"""



first,second,third = pieces
first+'::'+second+'::'+third
'::'.join(pieces)

"""
'a::b::guido'

"""


val.index(',') 
val.index('b')
val.index('g')
val.index('u')
#返回字符所在的位置，如果没有就报错

val.find
val.find('x') 
#返回字符所在的位置，如果没有输出-1


val.replace(',','::')  #  'a::b:: guido'
val.replace(',','')    #  'ab guido'
a,b,c = val.split(',')
 
"""

.count() 计算括号里符号出现的次数
.endswith('') .startswith('') 返回Ture 或false
join  '::'.join(pieces) 
index  返回字符所在的位置，如果没有就报错
find 返回第一个发现的字符所在的位置，如果没有输出-1
rfind 返回最后一个发现的字符所在的位置，如果没有输出-1
.replace(',','::') 用一个字符串替换另一个
strip(),rstrip(),lstrip()
split() 指定字符并拆分
lower upper 转换大小写
ljust、rjust 用空格或其他字符填充字符串的空白侧以返回符合最低宽度的字符串


"""

#-----------------------------------------------------------------------------
#正则表达式




#-----------------------------------------------------------------------------
#Pandas中矢量化的字符串函数


data = Series({'Dave':'dave@google.com',
        'steve':'steve@gmail.com',
        'rob':'rob@gmail.com',
        'wes':np.nan
        })

data.isnull()
data.str.contains('gmail')
matches = data.str.match('mail',flags=re.IGNORECASE)
matches.str[0]
matches.str.get(1)
data.str[:5]


str 矢量化的字符串方法

"""
data.str.cat(sep=',') 实现元素级的字符串连接操作，可指定分隔符号，
data.str.contains() 返回字符串是否含有指定模式的布尔数组
data.str.count('@')   模式出现的次数
data.str.endswith() data.str.startswith() 对各个元素执行
findall , 返回字符串的模式列表
get 获取字符串的第i个字符
join 根据指定的分隔符将Series中各元素的字符串联接起来
len 字符串长度
lower upper 相当于x.lower()
match 根据指定的正则表达式对各元素执行re.match
pad 在字符串左边右边或左右添加空白  
center 相当于  pad(side = 'both')
repeat 重复值 data.str.repeat(3) 相当于对各字符串执行x *3
slice 进行子串截取
data.str.split('@') 根据分隔符或正则表达式对字符串拆分
strip,rstrip,lstrip 相当于对各元素执行，x.strip()

"""




import json

db = json.load(open('ch07/foods-2011-10-03.json'))

len(db)

