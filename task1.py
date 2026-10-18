
# In[1]:


get_ipython().run_line_magic('pip', 'install seaborn')
get_ipython().run_line_magic('pip', 'install "numpy<1.23"')


# In[2]:


import pandas as pd
df = pd.read_csv("/Users/xiaziyue/Desktop/Mock/power_hourly.csv", parse_dates = ["timestamp"])
print(df.head())
df = df.drop_duplicates()
print("size:", df.size, "miss load:", df.load_mw.isna().sum())

print(df.isna().sum())
df = df.sort_values("timestamp")
df["timestamp"] = df["timestamp"].dt.tz_localize('Europe/Berlin',ambiguous = "infer").dt.tz_convert("UTC")
print(df.duplicated(subset = ["timestamp"]).sum())
d = df["timestamp"].diff()
print(d.value_counts())
lack = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq = "h").difference(df["timestamp"])
print(lack)
df = df.set_index("timestamp").asfreq("h")
df.loc[(df["load_mw"] > 100000) | (df["load_mw"]< 1),"load_mw" ] = None##这里就要改 后面shift了
#修改数值要用loc定位行列！连续【】可能改不到df
mark = df.isna().any(axis = 1)
print("num of columns with na:", mark.sum())
df = df.fillna(df.shift(168)).ffill().dropna()
print(df.isna().sum().sum())
print(df[df.isna().any(axis = 1)])







# In[ ]:





# In[3]:


import matplotlib.pyplot as plt
print(df.describe())
#temp_b_c impossible high T
df["temp_b_c"].plot()
df["temp_a_c"].plot()
df["temp_c_c"].plot()
plt.show()
##quit temp_b_c,usually low than average 

##price_eur_mwh  cannot be negative, max is too extreme
df["price_eur_mwh"] = df["price_eur_mwh"].clip(df["price_eur_mwh"].quantile(0.15),df["price_eur_mwh"].quantile(0.95))
#df["load_mw"] = df["load_mw"].clip(df["load_mw"].quantile(0.05),df["price_eur_mwh"].quantile(0.95))
print(df.describe())
df["price_eur_mwh"].hist(bins=50)#不太正常 有很多小于0的，删除还是clip
plt.show()
df["load_mw"].hist(bins=50)#right-tail 不确定是真实的还是错误，去掉exterme right tail：left tail很严重
plt.show()
df["load_mw"].plot.box()
plt.show()
df["gen_nuclear_mw"].hist(bins=50)#较为正常 cluster
plt.show()



# In[4]:


se = df["load_mw"].resample("D").mean().rolling(30).mean()
se.plot()
plt.show()##load 有周期性 一年，整体下降


# In[5]:


import seaborn as sns
df.index = df.index.tz_convert('Europe/Berlin')
map1 = df.groupby([df.index.hour,df.index.dayofweek])["load_mw"].mean().unstack()
sns.heatmap(map1)
plt.show()


# In[6]:


#print(df.index.month.tolist())
d1 = df[(df.index.month >= 3) & (df.index.month <= 5)]
spring = d1.groupby(d1.index.hour)["load_mw"].mean()
d2 = df[(df.index.month >= 6) & (df.index.month <= 8)]
summer = d2.groupby(d2.index.hour)["load_mw"].mean()
d3 = df[(df.index.month >= 9) & (df.index.month <= 11)]
autumn = d3.groupby(d3.index.hour)["load_mw"].mean()
d4 = df[(df.index.month >= 12) | (df.index.month <= 2)]
winter = d4.groupby(d4.index.hour)["load_mw"].mean()
al = pd.concat([spring,summer,autumn,winter], axis = 1)
al.columns = ["spring","summer","autumn","winter"]
al.plot()
plt.show()
dweekday = df[(df.index.dayofweek <= 4)]
weekdays = dweekday.groupby(dweekday.index.hour)["load_mw"].mean()
dweekend = df[(df.index.dayofweek > 4)]
weekend = dweekend.groupby(dweekend.index.hour)["load_mw"].mean()
all1 = pd.concat([weekdays, weekend], axis = 1)
all1.columns = ["weekdays", "weekend"]
all1.plot()
plt.show()


# In[ ]:





# #写法是先给每一行打上季节标签，再一次 groupby：
# se = pd.Series(df.index.month % 12 // 3, index=df.index).map({0: "winter", 1: "spring", 2: "summer", 3: "autumn"})
# s = df.groupby([df.index.hour, se])["load_mw"].mean().unstack()
# 我的想法：所有数据（train valid一起） 做特征（berlin time），在 D 日 10:00 知道吗？已经知道的：D+1 的小时、星期几、是否节假日，temp_forecast.csv 里 D+1 的预报。不知道的：temp_*_c（D+1 的实际温度），D+1 的 gen、price。 load 的 lag 能用到多近：如果对所有时间同样对待，大于38h。 如果我只是大约分成80% train 和20% valid，target对应的是预测的数值的时间，不存在info 泄漏到valid。不过如果给了test set（oos）评分，也需要拼在一起做特征（不然test就得重做一遍特征）。如果test和前面给的连续，test前几行不会有na，如果不连续，前几行会有na
# 
# .loc 和 .iloc 的区别
# 
# 	.loc	.iloc
# 按什么取	标签（这里是时间）	位置（第几行）
# 例子	X.loc[:"2024-03-31"]	X.iloc[:28000]
# 右端	包含	不包含
# 用在时间切分上	直接写日期，不容易出错	得先算出 3 月 31 日是第几行

# In[7]:


##notebook 里有一个习惯建议：会修改 df 的 cell（df = df....）最好只跑一次。
##想重来的话，就用 Kernel → Restart & Run All 从头跑一遍，不然很容易出现这种状态混乱的问题。
temp = pd.read_csv("/Users/xiaziyue/Desktop/Mock/temp_forecast.csv",parse_dates = ["target_time"])
print(temp.head())
temp.describe()
temp.isna().sum()
print("duplicated", temp.duplicated().sum())
temp["target_time"] = temp["target_time"].dt.tz_localize("Europe/Berlin",ambiguous = "infer")
print(temp["target_time"].diff().value_counts())##当地化后没有缺失数值！
temp = temp.set_index("target_time")
#df = pd.concat([df, temp], axis = 1)
df = df.join(temp, how = "outer")
df.head()
h = pd.get_dummies(df.index.hour, prefix = "h", drop_first = True).set_index(df.index)##get_dummies 返回df
dow = pd.get_dummies(df.index.dayofweek, prefix = "dow", drop_first = True).set_index(df.index)
m = pd.get_dummies(df.index.month, prefix = "m", drop_first = True).set_index(df.index)
year = pd.get_dummies(df.index.year, prefix = "year",drop_first = True).set_index(df.index)
off = (df.index.dayofweek >= 5).astype(int)
intes = h.mul(off, axis = 0).add_prefix("intesct")##要加入add_prefix 不然列名字一样
X = pd.concat([h, dow, m, year,intes], axis = 1)##列名字要不一样！
X.head()
lag48 = df["load_mw"].shift(48).rename("lag48")
lag168 = df["load_mw"].shift(168).rename("lag168")
y = df["load_mw"]
##X y 拼在一起dropna分割 不容易错位！
X = pd.concat([X,lag48,lag168,df["temp_fc_c"],y.rename("y")], axis = 1)

##index 是 DatetimeIndex 并且排好序的时候，.loc 会自动把 "2024-03-31" 这样的字符串解析成日期，再和 index 比较
#.loc 包含末尾
Xte = X.loc["2024-10-01":].drop(columns = "y")
yte = X.loc["2024-10-01":,"y"]
mark = mark.reindex(X.index, fill_value = False).astype(bool)
X = X.loc[~mark].dropna()
Xtr = X.loc[:"2024-03-31"].drop(columns = "y")
Xva = X.loc["2024-04-01":"2024-09-30"].drop(columns = "y")
ytr = X.loc[:"2024-03-31", "y"]
yva = X.loc["2024-04-01":"2024-09-30", "y"]


print(Xtr.head())
print(Xtr.shape, ytr.shape, Xva.shape,yva.shape, Xte.shape, yte.shape)



# In[8]:


from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
def train(X, target, alpha):
    pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha = alpha))])
    return pipe.fit(X, target)
    
diction ={}
for a in np.logspace(-4,4,50):##数据量大，ridge也要大
    m = train(Xtr, ytr, a)
    score = mean_squared_error(yva, m.predict(Xva)) ** 0.5
    diction[a] = score
    
best = min(diction, key = diction.get)
print("best alpha", best, "best RMSE", diction[best])

model = train(pd.concat([Xtr,Xva],ignore_index=True), pd.concat([ytr,yva],ignore_index=True), best)   
yte = model.predict(Xte)
#ouput = pd.Series(yte, index = Xte.index).reset_index()
#output.columns = ["timestamp", "load_mw"]
output = pd.DataFrame({"timestamp": Xte.index.tz_localize(None).strftime("%Y-%m-%d %H%M"), "load_mw": yte})
output.to_csv("/Users/xiaziyue/Desktop/Mock/forecast_20241001.csv", index = False)
    

# In[ ]:


##walk foward 和多种models（baseline，前一天的等，linear regression）：
def ev(y, p):
    return round(np.sqrt(((y - p) ** 2).mean())), round((abs(y - p) / y).mean() * 100, 2)

ps = {"seasonal_naive": [], "lr_base": [], "lr_full": []}
for mo in pd.period_range("2024-04", "2024-09", freq="M"):##每个month循环
    tr = D.loc[:str((mo.start_time - pd.Timedelta(days=1)).date())]
    va = D.loc[str(mo.start_time.date()):str(mo.end_time.date())]
    ps["seasonal_naive"].append(va.lag168)
       ps["lr_base"].append(pd.Series(LinearRegression().fit(tr[base], tr.y).predict(va[base]), index=va.index))
    ps["lr_full"].append(pd.Series(LinearRegression().fit(tr.drop(columns="y"), tr.y).predict(va.drop(columns="y")), index=va.index))
pv = {}
for k, v in ps.items():
    pv[k] = pd.concat(v)#按列对齐
yva = D.loc["2024-04-01":"2024-09-30", "y"]
pd.DataFrame({k: ev(yva, v) for k, v in pv.items()}, index=["RMSE", "MAPE%"])


#残差诊断：
#1. 残差 vs 预测值 2.残差自相关 3.分组：残差诊断：按小时、按日类
#残差r = yva - pv["lr_full"]
#pd.Series([r.autocorr(k) for k in range(1, 193)], index=range(1,193)).plot()

#最好/最坏的一周 .loc[string:string](前后都包括)
wk = r.groupby(r.index.to_period("W")).apply(lambda x: np.sqrt((x ** 2).mean()))#to_period 换成start_time/end_time
b, w = wk.idxmin(), wk.idxmax()
fig, ax = plt.subplots(2, 1, figsize=(14, 7))
yva.loc[str(b.start_time.date()):str(b.end_time.date())].plot(ax=ax[0], label="actual")
pv["lr_full"].loc[str(b.start_time.date()):str(b.end_time.date())].plot(ax=ax[0], label="forecast")
ax[0].set_title(f"best week {b}  RMSE {wk[b]:.0f}")
ax[0].legend()
yva.loc[str(w.start_time.date()):str(w.end_time.date())].plot(ax=ax[1], label="actual")
pv["lr_full"].loc[str(w.start_time.date()):str(w.end_time.date())].plot(ax=ax[1], label="forecast")
ax[1].set_title(f"worst week {w}  RMSE {wk[w]:.0f}")
ax[1].legend()
plt.tight_layout()
plt.show()


