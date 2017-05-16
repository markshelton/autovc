SYS_PATH = r'C:/Users/mark/Documents/GitHub/honours/dev/package/'
SAVE_PATH = r"C:/Users/mark/Documents/GitHub/honours/submissions/thesis-original"
LOAD_PATH = r"C:/Users/mark/Documents/GitHub/honours/dev/package/analysis/output/temp/output.db"

TINY_SIZE, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE  = 12, 16, 20, 30
plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_palette("colorblind")
rcParams['font.family'] = 'serif' 
rcParams['font.serif'] = 'CMU Serif' 
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = SMALL_SIZE #default font size
rcParams['axes.titlesize'] =SMALL_SIZE   # fontsize of the axes title
rcParams['axes.labelsize'] =MEDIUM_SIZE    # fontsize of the x and y labels
rcParams['xtick.labelsize'] =SMALL_SIZE    # fontsize of the tick labels
rcParams['ytick.labelsize'] =SMALL_SIZE   # fontsize of the tick labels
rcParams['legend.fontsize'] =SMALL_SIZE    # legend fontsize
rcParams['figure.titlesize'] =BIGGER_SIZE  # fontsize of the figure title
rcParams['figure.figsize'] = (8,3)
pd.set_option('precision',3)
outline_bars = dict(linewidth = 1.25, edgecolor = '.15')

def get_results(build, stage):

    df = pd.read_pickle(r"C:/Users/mark/Documents/GitHub/honours/dev/package/analysis/output/autoVC/{}/{}/log_results.pkl".format(build, stage))
    df = df.reset_index()
    try: df["Time"] = pd.to_numeric(df["mean_fit_time"],errors="coerce")
    except: print("Error: Time")
    try: df["Imputer"] = df["param_imputer__strategy"].map({"most_frequent": "Mode", "median": "Median", "mean": "Mean"})
    except: print("Error: Imputer")
    try: df["Transformer"] = df["param_transformer__func"].astype(str).map({"None":"None","<ufunc 'sqrt'>":"SQRT","<ufunc 'log1p'>": "Log1P"})
    except: print("Error: Transformer")
    try: df["Scaler"] = df['param_scaler'].apply(lambda x: str(x).split("(")[0])
    except: print("Error: Scaler")
    try:
        df["param_extractor__n_components_bin_20"] = df["param_extractor__n_components"] // 20
        df["Extractor"]=df['param_extractor__n_components_bin_20'].map({0 : "1-20", 1 : "21-40", 2: "41-60",3: "61-80", 4: "81-100"})
    except: print("Error: Extractor")
    try:
        df["Classifier"] = df["Classifier"].apply(lambda x: str(x).split("(")[0])
        df["Classifier"]=df['Classifier'].map({"LogisticRegression": "Logistic Regression", "RandomForestClassifier":"Random Forest", "DecisionTreeClassifier": "Decision Tree", "GaussianNB": "Naive Bayes", "MLPClassifier": "Artificial Neural Network", "KNeighborsClassifier": "K-Nearest Neighbors", "CalibratedClassifierCV": "Support Vector Machine"})
    except: print("Error: Classifier")
    try:
        df["label_date"] = df["label_slice"]
        df["label_date_str"] = df["label_date"].astype(str)
        df["feature_date"] = df["feature_slice"]
        df["feature_date_str"] = df["feature_date"].astype(str)
        df["forecast_window"] =  df["label_date"] - df["feature_date"]
        df["forecast_window_years"] = df["forecast_window"].apply(lambda x: x.days // 30) / 12
    except: print("Error: Dates")
    try: df["dataset_type"] = df["dataset_type"].map({"train":"Training Score","test":"Test Score"})
    except: print("Error: Learning Curve")
    try: df["label_type"] = df["label_type"].apply(lambda x: x.replace("_", " "))
    except: print("Error: Target Outcome")
    try:
        df["outcome_chance"] = df["label_name"].apply(lambda x: pd.Series(x).value_counts(normalize=True)[1])
    except: print("Error: Outcome")
    df["Params_str"] = df["Params"].astype(str)
    return df

def get_feature_values(df, total="Y_Pred"):

    def listify(col):
        if type(col.ix[0]) not in [list, np.ndarray, np.array, pd.Series]:
            return col.apply(lambda x: [x])
        else: return col

    def multip(row):
        for col in row.index:
            if col != "total" and type(row[col]) is list and len(row[col]) != row["total"][0]:
                row[col] = row[col] * row["total"][0]
        return row

    df["total"] = df[total].apply(len)
    df = df.apply(listify, axis=0)
    df = df.apply(multip, axis=1)
    df = df.drop("total",axis=1)
    df = df.apply(lambda x: list(chain.from_iterable(x)),axis=0)
    df = df.apply(pd.Series).T
    return df

def feature_function(df, func, group=None):
    df[["Y_Pred","Y_True"]] = df[["Y_Pred","Y_True"]].apply(lambda x: pd.to_numeric(x,errors="coerce"))
    if group: return df.groupby(group).apply(lambda x: func(x["Y_True"], x["Y_Pred"]))
    else: return func(df["Y_True"], df["Y_Pred"])

def divide_groups(x, totals):
    value = x.index.get_level_values(level=0)[0]
    x = x.apply(lambda x: x/float(totals.loc[value]))
    return x

#FORMATTING

def auto_label(ax, fmt='{:,.0f}', adjust=0):
    ymax_old = ax.get_ylim()[1]
    ax.set_ylim(ymax= ax.get_ylim()[1] * 1.1)
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            x=p.get_x()+p.get_width()/2.,
            y=np.nanmax([height,0]) + ymax_old * 0.02 + adjust,
            s=fmt.format(np.nanmax([height,0])),
            ha="center")
    ax.yaxis.set_ticks([])

def add_vertical_line(ax, value, label, color, linestyle="dashed"):
    ax.axvline(value, linestyle=linestyle, color=color)
    x_bounds = ax.get_xlim()
    xy_pos = (((value-x_bounds[0])/(x_bounds[1]-x_bounds[0])),1.01)
    ax.annotate(s=label, xy =xy_pos, xycoords='axes fraction', verticalalignment='right', horizontalalignment='right bottom', color=color)

def add_horizontal_line(ax, value, label, color, linestyle="dashed"):
    ax.axhline(value, linestyle=linestyle, color=color)
    y_bounds = ax.get_ylim()
    xy_pos = (1.01,((value-y_bounds[0])/(y_bounds[1]-y_bounds[0])))
    ax.annotate(s=label, xy =xy_pos, xycoords='axes fraction', verticalalignment='right', horizontalalignment='right bottom', color=color)

def add_line(ax, value, label, color, linestyle="dashed", orient="v"):
    if orient == "v": add_vertical_line(ax, value, label, color, linestyle=linestyle)
    elif orient == "h": add_horizontal_line(ax, value, label, color, linestyle=linestyle)
    else: raise ValueError("Valid Orientation not provided. Please provide 'v' or 'h'.")

def format_axis_ticks(f, axis="x", fmt="{:,}"):
    if type(f.axes[0]) is list: axes = f.axes[0]
    else: axes = [f.axes[0]]
    for ax in axes:
        if axis == "x": ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: fmt.format(int(x))))
        elif axis =="y": ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: fmt.format(int(x))))
        else: raise ValueError("Valid Axis not provided. Please provide 'x' or 'y'.")

def add_auc_to_legend(auc, ax, title):
    handles, labels = ax.get_legend_handles_labels()
    items = int(len(handles) / 2)
    handles = handles[0:items]
    labels = labels[0:items]
    auc = dict((str(k),v)for k,v in auc.items())
    labels = ["{} ({:,.3f})".format(label, auc[label]) for i, label in enumerate(labels)]
    plt.legend(loc="best", handles=handles, labels=labels, title=title)
